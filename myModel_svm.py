import torch
import torch.nn as nn
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm 
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import joblib
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"current device: {device}")

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 101)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = myModel().to(device)

model.load_state_dict(torch.load('myModel.pth')) 
print(model)

class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        self.features = original_model.features 
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return x

feature_extractor = FeatureExtractor(model).to(device)
feature_extractor.eval()

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Preparing dataset...")
svm_train_dataset = torchvision.datasets.Food101(root='./data', split='train', download=True, transform=test_transform)
test_dataset = torchvision.datasets.Food101(root='./data', split='test', download=True, transform=test_transform)

train_loader = DataLoader(svm_train_dataset, batch_size=128, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

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

print("Processing Train Data (myModel)...")
X_train, y_train = get_features(train_loader, feature_extractor, device)

print("Processing Test Data (myModel)...")
X_test, y_test = get_features(test_loader, feature_extractor, device)

print(f"Extracted Features Shape: {X_train.shape}") 


print("Training SVM Classifier...")
svm_clf = LinearSVC(max_iter=3000, C=0.1, verbose=1, dual=False) 
svm_clf.fit(X_train, y_train)

print("Evaluating SVM...")
y_pred = svm_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"SVM accuracy with myModel features: {acc * 100:.2f}%")

os.makedirs('models', exist_ok=True)
joblib.dump(svm_clf, 'models/mymodel_svm.pkl')
print("SVM Model saved to 'models/' directory")