import math
import os
import sys
import torch
import utils.misc as utils
import torch.nn.functional as F
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    cfg,
    args,
    datasize,
    start_time,
    writer=None,
):
    model.train()
    epoch_start_time = datetime.now()
    loss_type = cfg["loss"]

    psnr_list = []
    msssim_list = []
    coords_list = []  # 좌표 정보 저장

    for i, data in enumerate(dataloader):
        data = utils.to_cuda(data, device)
        
        # 좌표 정보 저장 (텐서 형태로 유지)
        norm_x = data["norm_x"]  # 이미 tensor
        norm_y = data["norm_y"]  # 이미 tensor
        coords = torch.stack([norm_x, norm_y], dim=1)  # [B, 2]
        coords_list.append(coords.detach().cpu())  # 텐서 기록용으로만 CPU로 변환
        
        # forward pass
        output_list = model(data)  # output is a list for the case that has multiscale
        additional_loss_item = {}
        if isinstance(output_list, dict):
            for k, v in output_list.items():
                if "loss" in k:
                    additional_loss_item[k] = v
            output_list = output_list["output_list"]
        target_list = [
            F.adaptive_avg_pool2d(data["img_gt"], x.shape[-2:]) for x in output_list
        ]
        loss_list = utils.loss_compute(output_list, target_list, loss_type)
        losses = sum(loss_list)
        if len(additional_loss_item.values()) > 0:
            losses = losses + sum(additional_loss_item.values())

        lr = utils.adjust_lr(optimizer, epoch, cfg["epoch"], i, datasize, cfg)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # compute psnr and msssim
        psnr_list.append(utils.psnr_fn(output_list, target_list))
        msssim_list.append(utils.msssim_fn(output_list, target_list))

        if i % cfg["print_freq"] == 0 or i == len(dataloader) - 1:
            train_psnr = torch.cat(psnr_list, dim=0)  # (batchsize, num_stage)
            train_psnr = torch.mean(train_psnr, dim=0)  # (num_stage)
            train_msssim = torch.cat(msssim_list, dim=0)  # (batchsize, num_stage)
            train_msssim = torch.mean(train_msssim.float(), dim=0)  # (num_stage)
            time_now_string = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            if not hasattr(args, "rank"):
                print_str = "[{}] Epoch[{}/{}], Step [{}/{}], lr:{:.2e} PSNR: {}, MSSSIM: {}".format(
                    time_now_string,
                    epoch + 1,
                    cfg["epoch"],
                    i + 1,
                    len(dataloader),
                    lr,
                    utils.RoundTensor(train_psnr, 2, False),
                    utils.RoundTensor(train_msssim, 4, False),
                )
                for k, v in additional_loss_item.items():
                    print_str += f", {k}: {v.item():.6g}"
                print(print_str, flush=True)

            elif args.rank in [0, None]:
                print_str = "[{}] Rank:{}, Epoch[{}/{}], Step [{}/{}], lr:{:.2e} PSNR: {}, MSSSIM: {}".format(
                    time_now_string,
                    args.rank,
                    epoch + 1,
                    cfg["epoch"],
                    i + 1,
                    len(dataloader),
                    lr,
                    utils.RoundTensor(train_psnr, 2, False),
                    utils.RoundTensor(train_msssim, 4, False),
                )
                print(print_str, flush=True)

    train_stats = {
        "train_psnr": train_psnr,
        "train_msssim": train_msssim,
    }
    if hasattr(args, "distributed") and args.distributed:
        train_stats = utils.reduce_dict(train_stats)

    # ADD train_PSNR TO TENSORBOARD
    if not hasattr(args, "rank"):
        h, w = output_list[-1].shape[-2:]
        writer.add_scalar(
            f"Train/PSNR_{h}X{w}", train_stats["train_psnr"][-1].item(), epoch + 1
        )
        writer.add_scalar(
            f"Train/MSSSIM_{h}X{w}", train_stats["train_msssim"][-1].item(), epoch + 1
        )
        writer.add_scalar("Train/lr", lr, epoch + 1)
        for k, v in additional_loss_item.items():
            writer.add_scalar(f"Train/{k}", v.item(), epoch + 1)
        
        # 좌표 정보 기록
        try:
            # 이미 CPU 텐서 리스트가 있으므로 이를 활용
            all_coords = torch.cat(coords_list, dim=0).numpy()
            writer.add_histogram('Train/X_Coordinates', all_coords[:, 0], epoch + 1)
            writer.add_histogram('Train/Y_Coordinates', all_coords[:, 1], epoch + 1)
            
            # 좌표 분포 시각화 (스캐터 플롯)
            if epoch % 10 == 0 and epoch > 0:  # 10 에폭마다 
                fig = plt.figure(figsize=(5, 5))
                plt.scatter(all_coords[:100, 0], all_coords[:100, 1], alpha=0.6)
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.xlabel('X Coordinate')
                plt.ylabel('Y Coordinate')
                plt.title(f'Sampled Light Field Coordinates (Epoch {epoch+1})')
                writer.add_figure('Train/LF_Coordinates', fig, epoch + 1)
        except Exception as e:
            print(f"Error recording coordinates: {e}")
            
        for (k, m) in model.named_modules():
            if isinstance(m, torch.nn.Module) and hasattr(m, "Lip_c"):
                writer.add_scalar(f"Stat/{k}_c", m.Lip_c[0].item(), epoch + 1)
                writer.add_scalar(f"Stat/{k}_w", m.abssum_max, epoch + 1)

    elif args.rank in [0, None] and writer is not None:
        h, w = output_list[-1].shape[-2:]
        writer.add_scalar(
            f"Train/PSNR_{h}X{w}", train_stats["train_psnr"][-1].item(), epoch + 1
        )
        writer.add_scalar(
            f"Train/MSSSIM_{h}X{w}", train_stats["train_msssim"][-1].item(), epoch + 1
        )
        writer.add_scalar("Train/lr", lr, epoch + 1)
        
        # 좌표 정보 기록
        try:
            all_coords = torch.cat(coords_list, dim=0).numpy()
            writer.add_histogram('Train/X_Coordinates', all_coords[:, 0], epoch + 1)
            writer.add_histogram('Train/Y_Coordinates', all_coords[:, 1], epoch + 1)
        except Exception as e:
            print(f"Error recording coordinates: {e}")
    
    epoch_end_time = datetime.now()
    print(
        "Time/epoch: \tCurrent:{:.2f} \tAverage:{:.2f}".format(
            (epoch_end_time - epoch_start_time).total_seconds(),
            (epoch_end_time - start_time).total_seconds() / (epoch + 1),
        )
    )

    return train_stats


@torch.no_grad()
def evaluate(model, dataloader, device, cfg, args, writer=None, save_image=False, epoch=0):
    val_start_time = datetime.now()
    model.eval()

    psnr_list = []
    msssim_list = []
    coords_list = []  # 좌표 정보 저장

    inf_time = []
    for i, data in enumerate(dataloader):
        data = utils.to_cuda(data, device)
        
        # 좌표 정보 저장 (텐서 형태로 유지)
        norm_x = data["norm_x"]  # 이미 tensor
        norm_y = data["norm_y"]  # 이미 tensor
        coords = torch.stack([norm_x, norm_y], dim=1)  # [B, 2]
        coords_list.append(coords.detach().cpu())  # 텐서 기록용으로만 CPU로 변환
        
        # forward pass
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        output_list = model(data)  # output is a list for the case that has multiscale
        end_event.record()
        
        
        if isinstance(output_list, dict):
            output_list = output_list["output_list"]  # ignore the loss in eval

        torch.cuda.synchronize()

        inf_time.append(start_event.elapsed_time(end_event) / 1000)

        target_list = [
            F.adaptive_avg_pool2d(data["img_gt"], x.shape[-2:]) for x in output_list
        ]

        # compute psnr and msssim
        psnr_list.append(utils.psnr_fn(output_list, target_list))
        msssim_list.append(utils.msssim_fn(output_list, target_list))

        # 이미지 저장 기능 추가
        if save_image and i < 5:  # 처음 5개 샘플만 저장
            save_dir = os.path.join(args.output_dir, 'val_images')
            os.makedirs(save_dir, exist_ok=True)
            
            # 각 샘플별로 처리
            for b in range(min(output_list[-1].size(0), 4)):  # 배치 중 최대 4개만
                x_val, y_val = norm_x[b].item(), norm_y[b].item()  # 텐서에서 값 추출
                
                # 출력 이미지
                output_img = output_list[-1][b].cpu()  # 가장 마지막 스케일(고해상도) 이미지 사용
                output_img = output_img.permute(1, 2, 0).numpy() * 255  # [H, W, 3], 0-255 범위로
                output_img = output_img.astype(np.uint8)
                
                # 타겟 이미지
                target_img = target_list[-1][b].cpu()
                target_img = target_img.permute(1, 2, 0).numpy() * 255
                target_img = target_img.astype(np.uint8)
                
                # OpenCV로 이미지 저장
                try:
                    import cv2
                    cv2.imwrite(os.path.join(save_dir, f'sample_{i}_{b}_x{x_val:.3f}_y{y_val:.3f}_output.png'), 
                                cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_dir, f'sample_{i}_{b}_x{x_val:.3f}_y{y_val:.3f}_target.png'), 
                                cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR))
                except ImportError:
                    # OpenCV가 없으면 PIL로 저장
                    from PIL import Image
                    Image.fromarray(output_img).save(
                        os.path.join(save_dir, f'sample_{i}_{b}_x{x_val:.3f}_y{y_val:.3f}_output.png'))
                    Image.fromarray(target_img).save(
                        os.path.join(save_dir, f'sample_{i}_{b}_x{x_val:.3f}_y{y_val:.3f}_target.png'))

        if i % cfg["print_freq"] == 0 or i == len(dataloader) - 1:
            val_psnr = torch.cat(psnr_list, dim=0)  # (batchsize, num_stage)
            val_psnr = torch.mean(val_psnr, dim=0)  # (num_stage)
            val_msssim = torch.cat(msssim_list, dim=0)  # (batchsize, num_stage)
            val_msssim = torch.mean(val_msssim.float(), dim=0)  # (num_stage)
            time_now_string = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            if not hasattr(args, "rank"):
                print_str = "[{}], Step [{}/{}], PSNR: {}, MSSSIM: {}".format(
                    time_now_string,
                    i + 1,
                    len(dataloader),
                    utils.RoundTensor(val_psnr, 2, False),
                    utils.RoundTensor(val_msssim, 4, False),
                )
                print(print_str, flush=True)

            elif args.rank in [0, None]:
                print_str = "[{}] Rank:{}, Step [{}/{}], PSNR: {}, MSSSIM: {}".format(
                    time_now_string,
                    args.rank,
                    i + 1,
                    len(dataloader),
                    utils.RoundTensor(val_psnr, 2, False),
                    utils.RoundTensor(val_msssim, 4, False),
                )
                print(print_str, flush=True)

    val_stats = {
        "val_psnr": val_psnr,
        "val_msssim": val_msssim,
    }
    if hasattr(args, "distributed") and args.distributed:
        val_stats = utils.reduce_dict(val_stats)
    val_end_time = datetime.now()
    print(
        "Time on evaluate: \t{:.2f}".format(
            (val_end_time - val_start_time).total_seconds()
        )
    )
    print(f"inference time: {sum(inf_time) / len(inf_time):.4f}s")

    # 텐서보드에 좌표별 성능 기록 (분포 확인)
    if args.rank in [0, None] and hasattr(args, "output_dir") and writer is not None:
        try:
            all_coords = torch.cat(coords_list, dim=0).numpy()
            writer.add_histogram('Eval/X_Coordinates', all_coords[:, 0], epoch)
            writer.add_histogram('Eval/Y_Coordinates', all_coords[:, 1], epoch)
            
            # 좌표 분포 시각화 (스캐터 플롯)
            fig = plt.figure(figsize=(5, 5))
            plt.scatter(all_coords[:100, 0], all_coords[:100, 1], alpha=0.6, c='red')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.title('Evaluation Light Field Coordinates')
            writer.add_figure('Eval/LF_Coordinates', fig, epoch)
        except Exception as e:
            print(f"Error recording eval coordinates: {e}")

    return val_stats
