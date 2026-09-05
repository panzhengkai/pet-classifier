import argparse
import os
import torch
import torch.optim as optim

from data.dataset import get_dataloaders
from models.model import create_model, LabelSmoothingCrossEntropy
from utils.metrics import set_seed, get_device, train_one_epoch, evaluate, plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description='Oxford-IIIT Pet 细粒度分类训练')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label Smoothing 系数，0 表示使用标准 CE')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.save_dir, exist_ok=True)

    # 加载数据
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        data_root=args.data_root, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed
    )

    # 创建模型
    model = create_model(num_classes=num_classes).to(device)

    # 定义损失函数（CrossEntropyLoss自带Softmax，不需要额外加Softmax层！）
    if args.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
        model_name = f'best_model_labelsmooth_{args.label_smoothing}.pth'
        exp_tag = f'Label Smoothing (ε={args.label_smoothing})'
    else:
        criterion = torch.nn.CrossEntropyLoss()
        model_name = 'best_model_baseline.pth'
        exp_tag = 'Baseline (标准 CE)'

    save_path = os.path.join(args.save_dir, model_name)

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # 训练参数
    best_val_acc = 0
    train_losses = []
    val_losses = []
    val_accs = []

    print(f"开始训练 {exp_tag}...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # 保存最好的模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    print(f"{exp_tag} 最佳验证准确率: {best_val_acc:.4f}")

    # 画训练曲线
    curves_path = os.path.join(args.save_dir, 'training_curves.png')
    plot_training_curves(train_losses, val_losses, val_accs, save_path=curves_path)


if __name__ == '__main__':
    main()
