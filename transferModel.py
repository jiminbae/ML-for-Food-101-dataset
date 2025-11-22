import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import matplotlib.pyplot as plt
import time

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

if __name__ == "__main__":
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 데이터셋 준비
    print("Preparing dataset...")
    if not os.path.exists('./data'):
        os.makedirs('./data')
        
    train_dataset = torchvision.datasets.Food101(root='./data', split='train', download=True, transform=train_transform)
    full_test_dataset = torchvision.datasets.Food101(root='./data', split='test', download=True, transform=test_transform)

    # Mini-Validation 데이터셋 분리 (전체의 20%만 사용)
    num_test = len(full_test_dataset)
    indices = list(range(num_test))
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * num_test)) 
    
    mini_val_dataset = Subset(full_test_dataset, indices[:split])
    
    print(f"전체 테스트 데이터: {num_test}장")
    print(f"학습 중 검증 데이터: {len(mini_val_dataset)}장 (속도를 위해 20%만 사용)")

    batch_size = 112
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(mini_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4) 
    full_test_loader = DataLoader(full_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4) 

    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    # 분류기 교체 (1000개 -> 101개)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 101)

    model = model.to(device)


    print("\n" + "="*60)
    print(f"          [ Model Architecture: {model.__class__.__name__} ]")
    print("="*60)
    print(model.classifier) 
    print("-" * 60)
    print("(Note: Full architecture print hidden to save space)")
    print("="*60 + "\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler('cuda')
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    epochs = 20
    best_acc = 0.0 

    train_losses = []
    val_accuracies = []

    print(f"Start Training for {epochs} epochs (with AMP)...")
    print("-" * 60)

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
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

        epoch_duration = time.time() - epoch_start_time
        epoch_mins = int(epoch_duration // 60)
        epoch_secs = int(epoch_duration % 60)
        
        print(f"[Epoch {epoch + 1}/{epochs}] Time: {epoch_mins}m {epoch_secs}s | Loss: {train_loss:.4f} | Mini-Val Acc: {mini_val_acc:.2f}% | LR: {current_lr:.1e}")

        train_losses.append(train_loss)
        val_accuracies.append(mini_val_acc)

        if mini_val_acc > best_acc:
            best_acc = mini_val_acc
            torch.save(model.state_dict(), 'transferModel.pth')
            print(f"  --> Best Model Saved ({best_acc:.2f}%)")
        
        scheduler.step()

        total_duration = time.time() - start_time
        total_mins = int(total_duration // 60)
        total_secs = int(total_duration % 60)

    print("-" * 60)
    print(f"학습 종료. 총 소요 시간: {total_mins}분 {total_secs}초")
    print(f"best accuracy(Mini-Val): {best_acc:.2f}%")

    # Full Evaluation
    model.load_state_dict(torch.load('transferModel.pth'))
    model.eval()
    
    correct = 0
    total = 0
    print("Final Evaluating (This may take a while)...")
    
    with torch.no_grad():
        for images, labels in full_test_loader: 
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    final_acc = 100 * correct / total
    print(f"최종 전체 테스트 정확도: {final_acc:.2f}%")

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='red')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), val_accuracies, label='Mini-Val Acc', color='blue')
    plt.title('Validation Accuracy (20% Subset)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('transferModel_result_graph.png')
    print("Graph saved as 'transferModel_result_graph.png'")
    
    plt.show()