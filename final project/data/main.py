import os
import pickle
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import clip

# 데이터 디렉토리 설정
data_dir = 'KSL_ACTION_VIDEO'

# 실제 존재하는 폴더 이름(두 자리 숫자) 가져오기
existing_folders = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
existing_class_indices = [int(folder) for folder in existing_folders]

# 클래스 레이블 로드
with open('class_label.p', 'rb') as f:
    class_labels = pickle.load(f)

# 실제 존재하는 클래스만 필터링
valid_class_labels = {k: v for k, v in class_labels.items() if k in existing_class_indices}
sorted_valid_classes = sorted(valid_class_labels.keys())[:10]  # 작은 숫자의 10개 클래스만 선택

# 클래스 매핑 (0부터 시작하는 새로운 인덱스)
class_to_idx = {cls: idx for idx, cls in enumerate(sorted_valid_classes)}
idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
num_classes = len(class_to_idx)

# 선택된 클래스 출력
print("선택된 클래스:", {k: valid_class_labels[k] for k in sorted_valid_classes})

def extract_frames(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    cap.release()
    if len(frames) < max_frames:
        frames.extend([np.zeros_like(frames[0])] * (max_frames - len(frames)))
    return np.array(frames)

device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP 모델 및 전처리기 로드
model, preprocess = clip.load("ViT-B/32", device=device)

def get_video_feature(frames):
    images = [preprocess(Image.fromarray(frame)).unsqueeze(0) for frame in frames]
    images = torch.cat(images).to(device)
    with torch.no_grad():
        image_features = model.encode_image(images)
    video_feature = image_features.mean(dim=0)
    return video_feature.cpu()

# 특징 벡터와 레이블을 저장할 리스트 초기화
features = []
labels = []

# 데이터 로드 및 특징 추출
print("데이터 로드 및 특징 추출 진행 중...")
for class_idx in tqdm(sorted_valid_classes, desc="Classes Processed"):
    folder_path = os.path.join(data_dir, f"{class_idx:02d}")
    for video_file in tqdm(os.listdir(folder_path), desc=f"Processing videos in class {class_idx:02d}", leave=False):
        video_path = os.path.join(folder_path, video_file)
        frames = extract_frames(video_path)
        video_feature = get_video_feature(frames)
        features.append(video_feature.numpy())
        labels.append(class_to_idx[class_idx])  # 매핑된 레이블 사용

features = torch.tensor(features)
labels = torch.tensor(labels)

# 학습 및 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Softmax 기반 PyTorch 모델 정의
class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SoftmaxClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)

input_dim = features.shape[1]
model = SoftmaxClassifier(input_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 학습
print("Softmax 모델 학습 중...")
epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train.to(device))
    loss = criterion(outputs, y_train.to(device))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 테스트
model.eval()
with torch.no_grad():
    outputs = model(X_test.to(device))
    predictions = torch.argmax(outputs, dim=1)
    accuracy = (predictions == y_test.to(device)).float().mean().item()
    print(f"Softmax 모델 정확도: {accuracy * 100:.2f}%")
