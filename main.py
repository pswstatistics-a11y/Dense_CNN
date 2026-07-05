# run_experiment.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import DenseCNN

def train(dims, epochs = 20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"학습 디바이스 : {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 망 억까 미러 우회 완료된 로컬 파일 로드
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    model = DenseCNN(dims=dims).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n분류 전 은닉층 차원: {dims}차원으로 학습 중")
    model.train()
    
    # 3에폭 가검증 루프 (체급을 키웠기 때문에 초반부터 정확도가 치솟음)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = (correct / total) * 100
        print(f"   ↳ [에폭 {epoch+1}/{epochs}] Loss: {epoch_loss:.4f} | 중간 정확도: {epoch_acc:.2f}%")

    save_file_name = f"model_dim_{dims}.pth"
    torch.save(model.state_dict(), save_file_name)
    print(f"{dims}차원 통제 실험군 가중치가 '{save_file_name}'으로 저장됨.")

if __name__ == "__main__":
    # 대조군 순차 실행: 고차원 대 저차원 fclayer
    train(dims=64, epochs=20)
    train(dims=8, epochs=20)
    print("학습 종료.")