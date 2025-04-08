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
import glob
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt

def load_yaml_as_dict(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def psnr(img1, img2):
    """이미지 간 PSNR 계산"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def main():
    parser = argparse.ArgumentParser('E-NeRV GT PSNR 평가', add_help=True)
    parser.add_argument('--cfg_path', type=str, required=True, help='설정 파일 경로')
    parser.add_argument('--checkpoint', type=str, required=True, help='모델 체크포인트 경로')
    parser.add_argument('--gt_path', type=str, default='/data/ysj/dataset/stanford_half/knights/images', help='GT 이미지 경로')
    parser.add_argument('--output_dir', type=str, default='psnr_results', help='결과 저장 경로')
    parser.add_argument('--indices', type=int, nargs='+', default=[72, 76, 80, 140, 144, 148, 208, 212, 216], help='평가할 인덱스 목록')
    parser.add_argument('--grid_size', type=int, default=17, help='라이트 필드 그리드 크기')
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
    
    # 변환 정의
    transform = transforms.ToTensor()
    
    # GT 이미지 로드
    gt_images = sorted(glob.glob(os.path.join(args.gt_path, '*.png')))
    grid_size = args.grid_size  # 17x17 그리드
    
    if len(gt_images) != grid_size * grid_size:
        print(f"경고: GT 이미지 수({len(gt_images)})가 예상({grid_size*grid_size})과 다릅니다!")
    
    # 결과 저장용 딕셔너리
    results = {
        'index': [],
        'u': [],
        'v': [],
        'norm_x': [],
        'norm_y': [],
        'psnr': []
    }
    
    # 각 인덱스에 대해 평가
    with torch.no_grad():
        for idx in args.indices:
            if idx >= len(gt_images):
                print(f"경고: 인덱스 {idx}가 GT 이미지 개수({len(gt_images)})를 초과합니다. 건너뜁니다.")
                continue
            
            # 그리드 좌표 계산 (u, v)
            v = idx // grid_size  # 행 (수직 방향)
            u = idx % grid_size   # 열 (수평 방향)
            
            # 좌표 정규화 (0~16 범위에서 -1~1 범위로)
            # 변환 공식: norm_value = (value / (grid_size - 1)) * 2 - 1
            norm_x = -((u / (grid_size - 1)) * 2 - 1)
            norm_y = (v / (grid_size - 1)) * 2 - 1
            
            print(f"처리 중: 인덱스 {idx}, 좌표 ({u}, {v}), 정규화 좌표 ({norm_x:.3f}, {norm_y:.3f})")
            
            # GT 이미지 로드
            gt_img_path = gt_images[idx]
            gt_img = Image.open(gt_img_path).convert("RGB")
            gt_tensor = transform(gt_img).unsqueeze(0).to(device)  # [1, 3, H, W]
            
            # 모델 입력 데이터 생성
            data = {
                'norm_x': torch.tensor([norm_x], dtype=torch.float32).to(device),
                'norm_y': torch.tensor([norm_y], dtype=torch.float32).to(device)
            }
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()  # 추론 시작 시간 기록
            
            # 모델 추론
            outputs = model(data)
            
            end_event.record()  # 추론 종료 시간 기록
            torch.cuda.synchronize()  # 모든 CUDA 연산 완료 대기
            elapsed_time = start_event.elapsed_time(end_event)  # 시간 측정 (밀리초 단위)
            print(f"모델 추론 시간: {elapsed_time:.2f} ms")
            # 가장 고해상도 이미지 가져오기
            if isinstance(outputs, dict):
                output_img = outputs['output_list'][-1]
            elif isinstance(outputs, list):
                output_img = outputs[-1]
            else:
                raise ValueError("지원되지 않는 출력 형식입니다.")
            
            # 출력 이미지 크기 조정 (GT와 동일하게)
            output_img = F.interpolate(output_img, size=gt_tensor.shape[2:], mode='bilinear', align_corners=False)
            
            # PSNR 계산
            psnr_value = psnr(output_img[0], gt_tensor[0]).item()
            
            # 결과 저장
            results['index'].append(idx)
            results['u'].append(u)
            results['v'].append(v)
            results['norm_x'].append(norm_x)
            results['norm_y'].append(norm_y)
            results['psnr'].append(psnr_value)
            
            # 이미지 저장
            output_np = output_img[0].permute(1, 2, 0).cpu().numpy() * 255
            output_np = output_np.astype(np.uint8)
            gt_np = gt_tensor[0].permute(1, 2, 0).cpu().numpy() * 255
            gt_np = gt_np.astype(np.uint8)
            
            # 결과 시각화 및 저장
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(gt_np)
            axes[0].set_title(f'GT 이미지 (인덱스: {idx})')
            axes[0].axis('off')
            
            axes[1].imshow(output_np)
            axes[1].set_title(f'생성 이미지 (PSNR: {psnr_value:.2f}dB)')
            axes[1].axis('off')
            
            fig.suptitle(f'좌표 ({u}, {v}) - 정규화 좌표 ({norm_x:.3f}, {norm_y:.3f})')
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f'compare_idx{idx}_u{u}_v{v}.png'))
            plt.close()
    
    # 전체 결과 출력
    print("\n===== 결과 요약 =====")
    print(f"{'인덱스':<10}{'(u, v)':<15}{'(norm_x, norm_y)':<25}{'PSNR(dB)':<10}")
    print("-" * 60)
    
    for i in range(len(results['index'])):
        print(f"{results['index'][i]:<10}({results['u'][i]}, {results['v'][i]}){'':5}"
              f"({results['norm_x'][i]:.3f}, {results['norm_y'][i]:.3f}){'':10}{results['psnr'][i]:.2f}")
    
    avg_psnr = sum(results['psnr']) / len(results['psnr'])
    print("-" * 60)
    print(f"평균 PSNR: {avg_psnr:.2f}dB")
    
    # 결과를 CSV 파일로 저장
    import csv
    with open(os.path.join(args.output_dir, 'psnr_results.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'u', 'v', 'norm_x', 'norm_y', 'psnr'])
        for i in range(len(results['index'])):
            writer.writerow([
                results['index'][i],
                results['u'][i],
                results['v'][i],
                results['norm_x'][i],
                results['norm_y'][i],
                results['psnr'][i]
            ])
    
    # 히트맵 시각화 준비 (모든 인덱스에 대한 PSNR 맵)
    heatmap_size = args.grid_size
    heatmap_data = np.zeros((heatmap_size, heatmap_size))
    
    # 계산된 위치에 PSNR 값 할당
    for i in range(len(results['index'])):
        u, v = results['u'][i], results['v'][i]
        heatmap_data[v, u] = results['psnr'][i]
    
    # PSNR 히트맵 시각화
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap_data, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='PSNR (dB)')
    plt.title('라이트 필드 PSNR 히트맵')
    plt.xlabel('u (가로)')
    plt.ylabel('v (세로)')
    
    # 평가한 위치 표시
    for i in range(len(results['index'])):
        u, v = results['u'][i], results['v'][i]
        plt.text(u, v, f'{results["psnr"][i]:.1f}', 
                 ha='center', va='center', color='w', fontweight='bold')
    
    plt.savefig(os.path.join(args.output_dir, 'psnr_heatmap.png'))
    plt.close()
    
    print(f"\n결과가 {args.output_dir} 디렉토리에 저장되었습니다.")

if __name__ == '__main__':
    main() 