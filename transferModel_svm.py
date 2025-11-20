import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

weights = EfficientNet_B0_Weights.DEFAULT

train_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Preparing dataset...")
train_dataset = torchvision.datasets.Food101(root='./data', split='train', download=True, transform=train_transform)
test_dataset = torchvision.datasets.Food101(root='./data', split='test', download=True, transform=test_transform)

train_dataset_clean = torchvision.datasets.Food101(root='./data', split='train', download=True, transform=test_transform)

batch_size = 128 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

svm_train_loader = DataLoader(train_dataset_clean, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# CNN Fine-Tuning
model = efficientnet_b0(weights=weights)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 101) # 101 classes

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20

print(f"Start CNN Fine-Tuning ({epochs} epochs)...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    # Train
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader: 
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_acc = 100 * correct / total
    print(f"[Epoch {epoch + 1}/{epochs}] Loss: {running_loss / len(train_loader):.4f} | Acc: {epoch_acc:.2f}%")

# defining Feature Extractor 
class EffNetFeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(EffNetFeatureExtractor, self).__init__()
        self.features = original_model.features
        self.avgpool = original_model.avgpool 
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x) # (Batch, 1280, 1, 1)
        x = self.flatten(x) # (Batch, 1280)
        return x

feature_extractor = EffNetFeatureExtractor(model).to(device)
feature_extractor.eval()

# CNN -> CPU Numpy
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

print("Processing Train Data for SVM (Clean Images)...")
X_train, y_train = get_features(svm_train_loader, feature_extractor, device)

print("Processing Test Data for SVM...")
X_test, y_test = get_features(test_loader, feature_extractor, device)

print(f"Extracted Features Shape: {X_train.shape}")

# SVM 학습
print("Training SVM Classifier...")

svm_clf = LinearSVC(max_iter=1000, C=1.0, verbose=1, dual=False) 
svm_clf.fit(X_train, y_train)

# evaluation
print("Evaluating SVM...")
y_pred = svm_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"EfficientNet + SVM Model Test Accuracy: {acc * 100:.2f}%")

joblib.dump(svm_clf, 'transferModel_svm.pkl')
torch.save(model.state_dict(), 'transferModel_svm_classifier.pth') 
print("Model saved")