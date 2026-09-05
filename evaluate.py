import argparse
import os
import torch

from data.dataset import get_dataloaders
from models.model import create_model
from utils.metrics import set_seed, get_device, evaluate, plot_confusion_matrix, compute_metrics
from utils.gradcam import GradCAM, show_cam_on_image


def parse_args():
    parser = argparse.ArgumentParser(description='模型评估与可视化')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=37)
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model_baseline.pth')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--no_confusion', action='store_true')
    parser.add_argument('--no_gradcam', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        data_root=args.data_root, batch_size=args.batch_size,
        num_workers=args.num_workers, download=False, seed=args.seed
    )

    # 加载最佳模型进行测试
    model = create_model(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"模型权重已加载: {args.checkpoint}")

    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"测试集准确率: {test_acc:.4f}")

    # 计算额外指标
    accuracy, macro_f1 = compute_metrics(test_labels, test_preds)
    print(f"Macro-F1: {macro_f1:.4f}")

    # 画混淆矩阵
    if not args.no_confusion:
        cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(test_labels, test_preds, num_classes=num_classes, save_path=cm_path)

    # Grad-CAM 可视化
    if not args.no_gradcam:
        # 选择模型最后一层卷积层
        target_layer = model.layer4[-1]
        grad_cam = GradCAM(model, target_layer)

        # 获取一张测试图片
        test_iter = iter(test_loader)
        images, labels = next(test_iter)
        img = images[0:1].to(device)  # 取第一张
        true_label = labels[0].item()

        # 预测
        with torch.no_grad():
            output = model(img)
            pred = torch.argmax(output, 1).item()

        # 生成热力图
        cam = grad_cam.generate(img, pred)

        # 显示并保存
        gradcam_path = os.path.join(args.output_dir, 'gradcam_example.png')
        show_cam_on_image(img[0], cam, pred, true_label, save_path=gradcam_path)

        print(f"预测类别: {pred}, 真实类别: {true_label}, "
              f"{'预测正确' if pred == true_label else '预测错误'}")


if __name__ == '__main__':
    main()
