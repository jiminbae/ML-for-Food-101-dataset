import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns


# CNN 학습용 (Augmentation)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 검증 및 SVM 학습용 (Clean)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if __name__ == "__main__":
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.exists('./data'):
        os.makedirs('./data')
    os.makedirs('models', exist_ok=True)

    # EfficientNet Training with AMP
    print("\n[Phase 1] Training CNN (EfficientNet-B0)...")
    
    # 데이터셋 로드
    train_dataset = torchvision.datasets.Food101(root='./data', split='train', download=True, transform=train_transform)
    full_test_dataset = torchvision.datasets.Food101(root='./data', split='test', download=True, transform=test_transform)
    
    # SVM용 Clean Train 데이터셋
    svm_train_dataset = torchvision.datasets.Food101(root='./data', split='train', download=True, transform=test_transform)

    # 20% Mini-Val
    num_test = len(full_test_dataset)
    indices = list(range(num_test))
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * num_test)) 
    mini_val_dataset = Subset(full_test_dataset, indices[:split])
    
    print(f"학습 중 검증 데이터: {len(mini_val_dataset)}장 (속도를 위해 20%만 사용)")

    # DataLoader
    batch_size = 112 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(mini_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # SVM 학습용 로더
    svm_train_loader = DataLoader(svm_train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(full_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 모델 설정
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    
    # 분류기 교체
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 101)
    model = model.to(device)

    print("\n" + "="*60)
    print(f"          [ Model Architecture: EfficientNet-B0 ]")
    print("="*60)
    print(model.classifier)
    print("-" * 60)
    print("(Note: Full architecture print hidden to save space)")
    print("="*60 + "\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler()
    
    epochs = 20
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_acc = 0.0
    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        
        # Mini-Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in valid_loader: 
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        mini_val_acc = 100 * correct / total
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {train_loss:.4f} | Mini-Val Acc: {mini_val_acc:.2f}% | LR: {current_lr:.1e}")

        train_losses.append(train_loss)
        val_accuracies.append(mini_val_acc)

        if mini_val_acc > best_acc:
            best_acc = mini_val_acc
            torch.save(model.state_dict(), 'models/transferModel_best.pth')
            print(f"  --> Best Model Saved ({best_acc:.2f}%)")
        
        scheduler.step()

    print("CNN Training Finished!")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='red')
    plt.title('Training Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), val_accuracies, label='Mini-Val Acc', color='blue')
    plt.title('Validation Accuracy (20% subset)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('transfer_training_graph.png')
    print("Training Graph saved as 'transfer_training_graph.png'")

    # Feature Extraction + SVM
    print("\n[Phase 2] Training SVM Classifier...")

    model.load_state_dict(torch.load('models/transferModel_best.pth'))
    
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

    def get_features(dataloader, model, device):
        features_list = []
        labels_list = []
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Extracting features"):
                images = images.to(device)
                outputs = model(images)
                features_list.append(outputs.cpu().numpy())
                labels_list.append(labels.numpy())
        return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)

    print("Processing Train Data for SVM...")
    X_train, y_train = get_features(svm_train_loader, feature_extractor, device)

    print("Processing Test Data for SVM...")
    X_test, y_test = get_features(test_loader, feature_extractor, device)

    print(f"Extracted Features Shape: {X_train.shape}")

    print("Training SVM Classifier...")
    svm_clf = LinearSVC(max_iter=1000, C=1.0, verbose=1, dual=False) 
    svm_clf.fit(X_train, y_train)

    print("Evaluating SVM...")
    y_pred = svm_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"EfficientNet + SVM Model Test Accuracy: {acc * 100:.2f}%")

    # SVM 모델 저장
    joblib.dump(svm_clf, 'models/transferModel_svm.pkl')
    print("SVM Model saved to 'models/' directory")

    print("Generating SVM Result Plots...")

    cm = confusion_matrix(y_test, y_pred)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    class_names = full_test_dataset.classes

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
    plt.savefig('transfer_svm_result.png')
    print("SVM Result Graphs saved as 'transfer_svm_result.png'")
    
    plt.show()