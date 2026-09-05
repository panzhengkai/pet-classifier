import numpy as np
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt


# 简化版Grad-CAM实现
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册钩子
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_image, target_class):
        self.model.eval()
        # 前向传播
        output = self.model(input_image)
        self.model.zero_grad()
        # 目标类的梯度
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)

        # 计算权重
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        # 加权激活
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().detach().numpy()
        # 归一化
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# 显示原图和热力图叠加
def show_cam_on_image(img_tensor, cam, pred, true_label, save_path='gradcam_example.png'):
    # 反标准化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = img.numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 1)

    # 叠加热力图
    heatmap = plt.cm.jet(cam)[:, :, :3]
    overlay = 0.5 * img + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    ax1.imshow(img)
    ax1.set_title('原图')
    ax1.axis('off')
    ax2.imshow(cam, cmap='jet')
    ax2.set_title('热力图')
    ax2.axis('off')
    ax3.imshow(overlay)
    ax3.set_title(f'叠加 (预测: {pred}, 真实: {true_label})')
    ax3.axis('off')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
