import torch
import torch.nn as nn
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader
import time 

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

if not os.path.exists('./models'):
    os.makedirs('./models')

if __name__ == "__main__":
    start_time = time.time()

    print("\nLoading Pre-trained EfficientNet...")

    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 101)

    model = model.to(device)

    try:
        model.load_state_dict(torch.load('transferModel.pth'))
        print("Pre-trained weights loaded from 'transferModel.pth'")
    except FileNotFoundError:
        print("'transferModel.pth' file not found. Please check the path.")
        exit()

    # Feature Extractor 정의
    class EffNetFeatureExtractor(nn.Module):
        def __init__(self, original_model):
            super(EffNetFeatureExtractor, self).__init__()
            self.features = original_model.features
            self.avgpool = original_model.avgpool 
            self.flatten = nn.Flatten()
            
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = self.flatten(x)
            return x

    feature_extractor = EffNetFeatureExtractor(model).to(device)
    feature_extractor.eval()

    # SVM용 Clean Data
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)), # EfficientNet은 224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Preparing dataset...")
    # SVM 학습용
    svm_train_dataset = torchvision.datasets.Food101(root='./data', split='train', download=True, transform=test_transform)
    # 최종 평가용
    test_dataset = torchvision.datasets.Food101(root='./data', split='test', download=True, transform=test_transform)

    class_names = test_dataset.classes

    batch_size = 128
    train_loader = DataLoader(svm_train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    def get_features(dataloader, model, device):
        features_list = []
        labels_list = []
        
        print("Extracting features...")
        with torch.no_grad():
            for images, labels in tqdm(dataloader):
                images = images.to(device)
                outputs = model(images)
                features_list.append(outputs.cpu().numpy())
                labels_list.append(labels.numpy())
                
        return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)

    print("\nTraining SVM Classifier...")

    print("Processing Train Data for SVM...")
    X_train, y_train = get_features(train_loader, feature_extractor, device)

    print("Processing Test Data for SVM...")
    X_test, y_test = get_features(test_loader, feature_extractor, device)

    print(f"Extracted Features Shape: {X_train.shape}")

    print("Training LinearSVC...")
    svm_clf = LinearSVC(max_iter=3000, C=1.0, verbose=1, dual=False) 
    svm_clf.fit(X_train, y_train)

    # evaluation
    print("Evaluating SVM...")
    y_pred = svm_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"EfficientNet + SVM Test Accuracy: {acc * 100:.2f}%")

    joblib.dump(svm_clf, 'models/transferModel_svm.pkl')
    print("SVM Model saved to 'models/transferModel_svm.pkl'")

    total_duration = time.time() - start_time
    total_mins = int(total_duration // 60)
    total_secs = int(total_duration % 60)
    print("-" * 60)
    print(f"학습 완료. 총 소요 시간: {total_mins}분 {total_secs}초")
    print("-" * 60)

    print("Generating Result Plots...")

    cm = confusion_matrix(y_test, y_pred)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)

    sorted_idx = np.argsort(class_accuracy)
    worst_10_idx = sorted_idx[:10]
    best_10_idx = sorted_idx[-10:]

    plt.figure(figsize=(15, 12))

    # Confusion Matrix
    plt.subplot(2, 1, 1)
    sns.heatmap(cm, cmap='Blues', xticklabels=False, yticklabels=False)
    plt.title(f'Confusion Matrix (Acc: {acc*100:.2f}%)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Best vs Worst Bar Chart
    plt.subplot(2, 1, 2)
    best_scores = class_accuracy[best_10_idx]
    best_names = [class_names[i] for i in best_10_idx]
    worst_scores = class_accuracy[worst_10_idx]
    worst_names = [class_names[i] for i in worst_10_idx]

    names = worst_names + best_names
    scores = np.concatenate([worst_scores, best_scores])
    colors = ['salmon']*10 + ['lightgreen']*10

    bars = plt.barh(range(len(names)), scores, color=colors)
    plt.yticks(range(len(names)), names)
    plt.xlim(0, 1.0)
    plt.axvline(0.5, color='gray', linestyle='--')
    plt.title('Top 10 Worst (Red) vs Best (Green) Performing Classes')
    plt.xlabel('Accuracy')

    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                 f'{scores[i]*100:.1f}%', va='center')

    plt.tight_layout()
    plt.savefig('transferModel_svm_result_graph.png')
    print("Graph saved as 'transferModel_svm_result_graph.png'")

    plt.show()