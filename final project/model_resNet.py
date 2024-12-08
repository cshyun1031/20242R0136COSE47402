import torch
import os
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# 1. 데이터 준비
DATA_DIR = "./data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
BATCH_SIZE = 32

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 2. 모델 정의
device = "cuda" if torch.cuda.is_available() else "cpu"
resnet_model = models.resnet50(pretrained=True)
NUM_CLASSES = len(train_dataset.classes)

# 분류기를 재정의
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, NUM_CLASSES)
resnet_model = resnet_model.to(device)

# 3. 손실 함수와 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet_model.fc.parameters(), lr=1e-4)

# 4. 학습 및 평가 함수 (ResNet과 동일)
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=(running_loss / len(train_loader)))
    return running_loss / len(train_loader)

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    loop = tqdm(test_loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loop.set_postfix(accuracy=(correct / total))
    return correct / total

# Accuracy 기록용 리스트
train_accuracies = []
val_accuracies = []

def calculate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# 5. 학습 루프
EPOCHS = 250 
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    
    # Training 단계
    train_loss = train(resnet_model, train_loader, optimizer, criterion, device)
    
    # Training Accuracy 계산
    train_acc = calculate_accuracy(resnet_model, train_loader, device)
    train_accuracies.append(train_acc)
    
    # Validation Accuracy 계산
    val_acc = calculate_accuracy(resnet_model, test_loader, device)
    val_accuracies.append(val_acc)
    
    print(f"Loss: {train_loss:.4f}, Train Accuracy: {train_acc * 100:.2f}%, Validation Accuracy: {val_acc * 100:.2f}%")

# 최종 테스트 정확도 출력
final_accuracy = evaluate(resnet_model, test_loader, device)
print(f"Final Test Accuracy: {final_accuracy * 100:.2f}%")

# Training/Validation Accuracy 곡선 시각화
plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS + 1), train_accuracies, label="Training Accuracy")
plt.plot(range(1, EPOCHS + 1), val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid()
plt.show()
