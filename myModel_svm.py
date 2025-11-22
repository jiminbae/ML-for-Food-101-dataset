import torch
import torch.nn as nn
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm 
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import joblib
import os
import matplotlib.pyplot as plt 
import seaborn as sns 
import time

# myModel
class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 101)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Feature Extractor
class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        self.features = original_model.features 
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return x

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
            
    X = np.concatenate(features_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    return X, y

if __name__ == "__main__":
    start_time = time.time()

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = myModel().to(device)

    try:
        model.load_state_dict(torch.load('models/myModel.pth')) 
        print("myModel weights loaded successfully.")
    except FileNotFoundError:
        print("'models/myModel.pth' not found.")

    print("\n" + "="*60)
    print("               [ Model Architecture Summary ]               ")
    print("="*60)
    print(model) 
    print("="*60 + "\n")

    feature_extractor = FeatureExtractor(model).to(device)
    feature_extractor.eval()

    # SVM용 Clean Data
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print("Preparing dataset...")
    svm_train_dataset = torchvision.datasets.Food101(root='./data', split='train', download=True, transform=test_transform)
    test_dataset = torchvision.datasets.Food101(root='./data', split='test', download=True, transform=test_transform)
    
    class_names = test_dataset.classes

    train_loader = DataLoader(svm_train_dataset, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    # 특징 추출 및 SVM 학습
    print("Processing Train Data (myModel)...")
    X_train, y_train = get_features(train_loader, feature_extractor, device)

    print("Processing Test Data (myModel)...")
    X_test, y_test = get_features(test_loader, feature_extractor, device)

    print(f"Extracted Features Shape: {X_train.shape}") 

    print("Training SVM Classifier...")
    svm_clf = LinearSVC(max_iter=3000, C=0.1, verbose=1, dual=False) 
    svm_clf.fit(X_train, y_train)

    # evaluate SVM
    print("Evaluating SVM...")
    y_pred = svm_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"SVM accuracy with myModel features: {acc * 100:.2f}%")

    total_duration = time.time() - start_time
    total_mins = int(total_duration // 60)
    total_secs = int(total_duration % 60)
    print("-" * 60)
    print(f"학습 완료. 총 소요 시간: {total_mins}분 {total_secs}초")
    print("-" * 60)

    os.makedirs('models', exist_ok=True)
    joblib.dump(svm_clf, 'models/mymodel_svm.pkl')
    print("SVM Model saved to 'models/mymodel_svm.pkl'")

    # Result Analysis
    print("Generating Result Plots...")

    cm = confusion_matrix(y_test, y_pred)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)

    sorted_idx = np.argsort(class_accuracy)
    worst_10_idx = sorted_idx[:10]
    best_10_idx = sorted_idx[-10:]

    plt.figure(figsize=(15, 12))

    # Plot 1: Confusion Matrix
    plt.subplot(2, 1, 1)
    sns.heatmap(cm, cmap='Blues', xticklabels=False, yticklabels=False)
    plt.title(f'Confusion Matrix (Acc: {acc*100:.2f}%)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Plot 2: Best vs Worst Bar Chart
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
    plt.show()
    plt.savefig('myModel_svm_result_graph.png')
    print("Graphs saved as 'myModel_svm_result_graph.png'")