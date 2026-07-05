# viisualize.py
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import DenseCNN # 마스터와 정립한 구조 복사

def render_manifold_space(dim_size, model_path_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 텐서 전처리 공급망 (Z-Score 표준화)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 검증용 데이터 로드
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # 모델 선언 및 훈련 완료된 가중치 파일 이식
    model = DenseCNN(dims=dim_size).to(device)
    try:
        model.load_state_dict(torch.load(model_path_name, map_location=device))
        print(f"로드 성공 '{model_path_name}'")
    except Exception as e:
        print(f"⚠️ [주의] 가중치 파일 로드 실패({e}). 초기화 난수 상태로 진행합니다.")
        
    model.eval() # 가중치 동결 (미분 중지)

    all_latents = []
    all_labels = []
    
    # 검증 정확도 측정
    correct_predictions = 0
    total_samples = 0
    
    print("검증 데이터셋 연산 중...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels_dev = images.to(device), labels.to(device)
            
            # 순방향 전파
            outputs, latent_features = model(images) 
            
            # 최종 분류 정확도 누적 계산
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels_dev.size(0)
            correct_predictions += (predicted == labels_dev).sum().item()
            
            all_latents.append(latent_features.cpu().numpy())
            all_labels.append(labels.numpy())
                
    # 최종 검증 정확도 콘솔 정산 출력
    final_test_accuracy = (correct_predictions / total_samples) * 100
    print(f"[최종 검증 스코어 정산] 모델 차원 크기: {dim_size}차원")
    print(f"   ↳ 최종 테스트 정확도 (Test Accuracy): {final_test_accuracy:.2f}%")
    
    latents = np.concatenate(all_latents, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print(f"고차원 특징 공간 ➔ 2차원 시각 평면 투영 (t-SNE)")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    embedded_space = tsne.fit_transform(latents)
    
    # 시각화
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plt.figure(figsize=(12, 10))
    for i in range(10):
        indices = np.where(labels == i)
        plt.scatter(embedded_space[indices, 0], embedded_space[indices, 1], label=classes[i], alpha=0.5, s=3)
        
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.title(f"Feature Space Geometry (Dimension Size: {dim_size} / Test Acc: {final_test_accuracy:.2f}%)", fontsize=14)
    
    output_filename = f"./manifold_report_dim_{dim_size}.png"
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"시각화 이미지 파일 '{output_filename}' 저장됨\n")

if __name__ == "__main__":
    # 1. 64차원 대조군 모델 정산
    render_manifold_space(dim_size=64, model_path_name="model_dim_64.pth")
    
    # 2. 8차원 압축군 모델 정산 (서로 오염 없이 깔끔하게 독립 처리)
    render_manifold_space(dim_size=8, model_path_name="model_dim_8.pth")