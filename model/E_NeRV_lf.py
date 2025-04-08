import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributions as dist
from .model_utils import ActivationLayer, NormLayer, PositionalEncoding, gradient
from .NeRV import NeRV_MLP, NeRVBlock, Conv_Up_Block
from einops import rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = heads * dim_head
        project_out = not(heads==1 and dim_head==dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0., prenorm=False):
        super(TransformerBlock, self).__init__()
        if prenorm:
            self.attn = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
            self.ffn = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        else:
            self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)
    def forward(self, x):
        x = self.attn(x) + x
        x = self.ffn(x) + x
        return x


class E_NeRV_Generator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # t mapping
        self.pe_t = PositionalEncoding(
            pe_embed_b=cfg['pos_b'], pe_embed_l=cfg['pos_l']
        )

        stem_dim_list = [int(x) for x in cfg['stem_dim_num'].split('_')]
        self.fc_h, self.fc_w, self.fc_dim = [int(x) for x in cfg['fc_hw_dim'].split('_')]
        self.block_dim = cfg['block_dim']

        mlp_dim_list = [self.pe_t.embed_length] + stem_dim_list + [self.block_dim]
        self.stem_t = NeRV_MLP(dim_list=mlp_dim_list, act=cfg['act'])
        
        # xy 좌표를 결합하기 위한 추가 프로젝션
        self.xy_proj = nn.Linear(self.block_dim * 2, self.block_dim)
        self.manipulate_proj = nn.Linear(128 * 2, 128)

        # xy mapping
        xy_coord = torch.stack( 
            torch.meshgrid(
                torch.arange(self.fc_h) / self.fc_h, torch.arange(self.fc_w) / self.fc_w
            ), dim=0
        ).flatten(1, 2)  # [2, h*w]
        self.xy_coord = nn.Parameter(xy_coord, requires_grad=False)
        self.pe_xy = PositionalEncoding(
            pe_embed_b=cfg['xypos_b'], pe_embed_l=cfg['xypos_l']
        )
        
        self.stem_xy = NeRV_MLP(dim_list=[2 * self.pe_xy.embed_length, self.block_dim], act=cfg['act'])
        self.trans1 = TransformerBlock(
            dim=self.block_dim, heads=1, dim_head=64, mlp_dim=cfg['mlp_dim'], dropout=0., prenorm=False
        )
        self.trans2 = TransformerBlock(
            dim=self.block_dim, heads=8, dim_head=64, mlp_dim=cfg['mlp_dim'], dropout=0., prenorm=False
        )
        if self.block_dim == self.fc_dim:
            self.toconv = nn.Identity()
        else:
            self.toconv = NeRV_MLP(dim_list=[self.block_dim, self.fc_dim], act=cfg['act'])
        
        # BUILD CONV LAYERS
        self.layers, self.head_layers, self.t_layers, self.norm_layers = [nn.ModuleList() for _ in range(4)]
        ngf = self.fc_dim
        for i, stride in enumerate(cfg['stride_list']):
            if i == 0:
                # expand channel width at first stage
                new_ngf = int(ngf * cfg['expansion'])
            else:
                # change the channel width for each stage
                new_ngf = max(ngf // (1 if stride == 1 else cfg['reduction']), cfg['lower_width'])
            
            self.t_layers.append(NeRV_MLP(dim_list=[128, 2*ngf], act=cfg['act']))
            self.norm_layers.append(nn.InstanceNorm2d(ngf, affine=False))
            
            if i == 0:
                self.layers.append(Conv_Up_Block(ngf=ngf, new_ngf=new_ngf, stride=stride, bias=cfg['bias'], norm=cfg['norm'], act=cfg['act'], conv_type=cfg['conv_type']))
            else:
                self.layers.append(NeRVBlock(ngf=ngf, new_ngf=new_ngf, stride=stride, bias=cfg['bias'], norm=cfg['norm'], act=cfg['act'], conv_type=cfg['conv_type']))
            ngf = new_ngf

            # build head classifier, upscale feature layer, upscale img layer 
            head_layer = [None]
            if cfg['sin_res']:
                if i == len(cfg['stride_list']) - 1:
                    head_layer = nn.Conv2d(ngf, 3, 1, 1, bias=cfg['bias'])
                else:
                    head_layer = None
            else:
                head_layer = nn.Conv2d(ngf, 3, 1, 1, bias=cfg['bias'])
            self.head_layers.append(head_layer)
        self.sigmoid = cfg['sigmoid']

        self.T_num = 20
        self.pe_t_manipulate = PositionalEncoding(pe_embed_b=cfg['pos_b_tm'], pe_embed_l=cfg['pos_l_tm'])
        self.t_branch = NeRV_MLP(dim_list=[self.pe_t_manipulate.embed_length, 128, 128], act=cfg['act'])

        self.loss = cfg['additional_loss'] if cfg.__contains__('additional_loss') else None
        self.loss_w = cfg['additional_loss_weight'] if cfg.__contains__('additional_loss_weight') else 1.0
        self.mse = nn.MSELoss()
    
    def fuse_t(self, x, t):
        # x: [B, C, H, W], normalized among C
        # t: [B, 2* C]
        f_dim = t.shape[-1] // 2
        gamma = t[:, :f_dim]
        beta = t[:, f_dim:]

        gamma = gamma[..., None, None]
        beta = beta[..., None, None]
        out = x * gamma + beta
        return out

    def forward_impl(self, x, y):
        # x, y: [B]

        # x와 y를 각각 임베딩한 후 concatenate
        x_emb = self.pe_t(x)  # [B, embed_length]
        y_emb = self.pe_t(y)  # [B, embed_length]
        
        # 각각 임베딩한 후 stem 네트워크 통과
        x_feat = self.stem_t(x_emb)  # [B, block_dim]
        y_feat = self.stem_t(y_emb)  # [B, block_dim]
        
        # x와 y 특성을 결합하고 원래 차원으로 프로젝션
        xy_feat = torch.cat([x_feat, y_feat], dim=-1)  # [B, 2*block_dim]
        t_emb = self.xy_proj(xy_feat)  # [B, block_dim]
        
        # t_manipulate도 비슷한 방식으로 처리
        x_manipulate = self.pe_t_manipulate(x)
        y_manipulate = self.pe_t_manipulate(y)
        x_branch = self.t_branch(x_manipulate)  # [B, 128]
        y_branch = self.t_branch(y_manipulate)  # [B, 128]
        xy_manipulate = torch.cat([x_branch, y_branch], dim=-1)  # [B, 2*128]
        t_manipulate = self.manipulate_proj(xy_manipulate)  # [B, 128]

        xy_coord = self.xy_coord
        x_coord = self.pe_xy(xy_coord[0])    # [h*w, C]
        y_coord = self.pe_xy(xy_coord[1])    # [h*w, C]
        xy_emb = torch.cat([x_coord, y_coord], dim=1)
        xy_emb = self.stem_xy(xy_emb).unsqueeze(0).expand(x_feat.shape[0], -1, -1)  # [B, h*w, L]

        xy_emb = self.trans1(xy_emb)
        # fuse t into xy map
        t_emb_list = [t_emb for i in range(xy_emb.shape[1])]
        t_emb_map = torch.stack(t_emb_list, dim=1)  # [B, h*w, block_dim]
        emb = xy_emb * t_emb_map
        emb = self.toconv(self.trans2(emb))

        emb = emb.reshape(emb.shape[0], self.fc_h, self.fc_w, emb.shape[-1])
        emb = emb.permute(0, 3, 1, 2)
        output = emb

        out_list = []
        for layer, head_layer, t_layer, norm_layer in zip(self.layers, self.head_layers, self.t_layers, self.norm_layers):
            # t_manipulate
            output = norm_layer(output)
            t_feat = t_layer(t_manipulate)  # 이제 차원이 맞음
            output = self.fuse_t(output, t_feat)
            # conv
            output = layer(output) 
            if head_layer is not None:
                img_out = head_layer(output)
                # normalize the final output iwth sigmoid or tanh function
                img_out = torch.sigmoid(img_out) if self.sigmoid else (torch.tanh(img_out) + 1) * 0.5
                out_list.append(img_out)

        return out_list
    
    def forward(self, data):
        # norm_x와 norm_y를 사용하도록 변경
        x = data['norm_x']  # [B]
        y = data['norm_y']  # [B]
        batch_size = x.shape[0]

        output_list = self.forward_impl(x, y)  # a list containing [B or 2B, 3, H, W]

        if self.loss and self.training:
            b, c, h, w = output_list[-1].shape
            # NO USE
            grad_loss = 0.0
            return {
                "loss": grad_loss * self.loss_w,
                "output_list": output_list,
            }
        
        return output_list
