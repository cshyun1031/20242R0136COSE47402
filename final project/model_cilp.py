import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import clip
from torch import nn, optim
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# 1. 데이터 준비
DATA_DIR = "./data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
BATCH_SIZE = 32

# 2. 데이터 전처리 (CLIP 전처리 적용)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ImageFolder로 데이터 로드
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=preprocess)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=preprocess)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. 분류기를 추가한 모델 정의
class SignLanguageClassifier(nn.Module):
    def __init__(self, clip_model, num_classes):
        super(SignLanguageClassifier, self).__init__()
        self.clip_model = clip_model
        # Classifier 생성 시, clip_model의 dtype을 확인하여 일치시킴
        self.classifier = nn.Linear(clip_model.visual.output_dim, num_classes)
        self.classifier = self.classifier.to(torch.float32)  # FP32로 강제 변환

    def forward(self, images):
        with torch.no_grad():
            # 입력 이미지를 FP32로 변환
            image_features = self.clip_model.encode_image(images.to(torch.float32))
        return self.classifier(image_features.to(torch.float32))  # classifier에 FP32로 전달




NUM_CLASSES = len(train_dataset.classes)
model = SignLanguageClassifier(model, NUM_CLASSES).to(device)

# 4. 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)

# 5. 학습 및 평가 함수
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


# 4. 데이터 전처리 - Data Augmentation 추가
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

EPOCHS = 200

# 데이터셋에 전처리 적용
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 5. 옵티마이저와 스케줄러
optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=EPOCHS)



# 6. 학습 루프 - Gradient Clipping 및 스케줄러 적용
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    model.train()
    running_loss = 0.0

    loop = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()  # 스케줄러 업데이트

        running_loss += loss.item()
        loop.set_postfix(loss=(running_loss / len(train_loader)))

    # Training/Validation Accuracy 계산
    train_acc = calculate_accuracy(model, train_loader, device)
    val_acc = calculate_accuracy(model, test_loader, device)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"Loss: {running_loss:.4f}, Train Accuracy: {train_acc * 100:.2f}%, Validation Accuracy: {val_acc * 100:.2f}%")

# 최종 테스트 정확도 출력
final_accuracy = evaluate(model, test_loader, device)
print(f"Final Test Accuracy (Optimized): {final_accuracy * 100:.2f}%")

# Training/Validation Accuracy 곡선 시각화
plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS + 1), train_accuracies, label="Training Accuracy")
plt.plot(range(1, EPOCHS + 1), val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy (Optimized)")
plt.legend()
plt.grid()
plt.show()
