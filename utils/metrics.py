import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score


# 固定随机种子，保证每次跑的结果一样
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    # 检测GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    return device


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # 重要：清零梯度（防止梯度累积）
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()  # 重要：切换到评估模式（关闭Dropout和BN的随机性）
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 重要：不计算梯度，节省显存
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def plot_training_curves(train_losses, val_losses, val_accs, save_path='training_curves.png'):
    # 画训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss曲线')

    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('验证准确率曲线')
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_confusion_matrix(all_labels, all_preds, num_classes=37, save_path='confusion_matrix.png'):
    # 画混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=range(37), yticklabels=range(37))
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title('混淆矩阵 (37类)')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    # 找出最容易被混淆的类别对
    # 忽略对角线，找最大的非对角线值
    np.fill_diagonal(cm, 0)
    most_confused = np.unravel_index(np.argmax(cm), cm.shape)
    print(f"最容易被混淆的类别对: {most_confused[0]} <-> {most_confused[1]}, 混淆次数: {cm[most_confused]}")
    return most_confused


def compute_metrics(all_labels, all_preds):
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return accuracy, macro_f1
