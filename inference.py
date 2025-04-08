import argparse
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import yaml
import sys
from model import model_dict

def load_yaml_as_dict(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_image(tensor, path):
    """텐서를 이미지로 변환하여 저장"""
    # [C, H, W] -> [H, W, C]
    img = tensor.permute(1, 2, 0).cpu().numpy()
    # 0-1 값 범위를 0-255로 변환
    img = (img * 255).astype(np.uint8)
    # 이미지 저장
    Image.fromarray(img).save(path)

def main():
    parser = argparse.ArgumentParser('E-NeRV Inference', add_help=True)
    parser.add_argument('--cfg_path', type=str, required=True, help='설정 파일 경로')
    parser.add_argument('--checkpoint', type=str, required=True, help='모델 체크포인트 경로')
    parser.add_argument('--output_dir', type=str, default='results', help='출력 이미지 저장 경로')
    parser.add_argument('--coords', type=str, nargs='+', default=[], help='x,y 좌표 쌍 (예: 0.2,0.3 0.5,0.7)')
    parser.add_argument('--grid', action='store_true', help='격자 패턴으로 좌표 생성')
    parser.add_argument('--grid_size', type=int, default=5, help='격자 크기 (NxN)')
    parser.add_argument('--norm_range', type=str, default='-1,1', help='좌표 정규화 범위 (기본값: -1,1 / 옵션: 0,1)')
    args = parser.parse_args()

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # 설정 파일 로드
    cfg = load_yaml_as_dict(args.cfg_path)
    
    # 모델 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"장치: {device}")
    
    model = model_dict[cfg['model']['model_name']](cfg=cfg['model'])
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    try:
        model.load_state_dict(checkpoint['model'])
    except:
        # 분산 학습 모델인 경우
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            if k.startswith('module.'):
                name = k[7:]  # 'module.' 제거
            else:
                name = k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    
    model.to(device)
    model.eval()
    
    # 정규화 범위 파싱
    norm_min, norm_max = map(float, args.norm_range.split(','))
    print(f"좌표 정규화 범위: [{norm_min}, {norm_max}]")
    
    # 좌표 목록 생성
    coords_list = []
    
    if args.grid:
        # 격자 패턴으로 좌표 생성
        step = 1.0 / (args.grid_size - 1) if args.grid_size > 1 else 0.5
        for i in range(args.grid_size):
            for j in range(args.grid_size):
                # 0~1 범위의 좌표 생성
                x_01 = i * step
                y_01 = j * step
                
                # 정규화 범위로 변환 (0~1 -> norm_min~norm_max)
                x = x_01 * (norm_max - norm_min) + norm_min
                y = y_01 * (norm_max - norm_min) + norm_min
                
                coords_list.append((x, y, x_01, y_01))  # 변환 전후 좌표 저장
    else:
        # 사용자 지정 좌표
        for coord_str in args.coords:
            x_01, y_01 = map(float, coord_str.split(','))
            
            # 입력된 좌표가 0~1 범위라고 가정하고 정규화 범위로 변환
            x = x_01 * (norm_max - norm_min) + norm_min
            y = y_01 * (norm_max - norm_min) + norm_min
            
            coords_list.append((x, y, x_01, y_01))
    
    # 좌표가 없으면 기본값 사용
    if not coords_list:
        x_01, y_01 = 0.5, 0.5  # 중앙 좌표 (0~1 범위)
        x = x_01 * (norm_max - norm_min) + norm_min
        y = y_01 * (norm_max - norm_min) + norm_min
        coords_list = [(x, y, x_01, y_01)]
    
    print(f"생성할 이미지 수: {len(coords_list)}")
    
    # 각 좌표에 대해 이미지 생성
    with torch.no_grad():
        for i, (x, y, x_01, y_01) in enumerate(coords_list):
            # 입력 데이터 생성
            norm_x = torch.tensor([x], dtype=torch.float32).to(device)
            norm_y = torch.tensor([y], dtype=torch.float32).to(device)
            
            data = {
                'norm_x': norm_x,
                'norm_y': norm_y
            }
            
            # 모델 추론
            outputs = model(data)
            
            # 가장 고해상도 이미지 가져오기 (리스트의 마지막 요소)
            if isinstance(outputs, dict):
                output_img = outputs['output_list'][-1]
            elif isinstance(outputs, list):
                output_img = outputs[-1]
            else:
                raise ValueError("지원되지 않는 출력 형식입니다.")
            
            # 배치 크기가 1이므로 첫 번째 이미지 사용
            output_img = output_img[0]
            
            # 이미지 저장 (파일명에는 0~1 범위의 좌표 사용)
            save_path = os.path.join(args.output_dir, f'img_x{x_01:.3f}_y{y_01:.3f}_norm{x:.3f}_{y:.3f}.png')
            save_image(output_img, save_path)
            print(f"이미지 저장됨: {save_path}")

if __name__ == '__main__':
    main() 