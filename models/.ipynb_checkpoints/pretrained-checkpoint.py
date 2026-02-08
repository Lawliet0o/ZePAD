import os
import torch
from pathlib import Path
import torch.nn as nn
from models.resnet import resnet18



def load_pretrained_model(args):
    if args.pre_dataset == 'cifar10':
        num_classes = 10
        pretrained_path = os.path.join('pretrained', 'cifar10', str(args.pretrained))
        encoder_path = [Path(pretrained_path) / ckpt for ckpt in os.listdir(Path(pretrained_path)) if ckpt.endswith(".ckpt")][0]
        model = resnet18(num_classes=num_classes)
        checkpoint = torch.load(encoder_path)
        state_dict = checkpoint['state_dict']

        new_ckpt = dict()
        for k, value in state_dict.items():
            if k.startswith('backbone'):
                new_ckpt[k.replace('backbone.', '')] = value
            elif k.startswith('classifier'):
                new_ckpt[k.replace('classifier', 'fc')] = value
            else:
                new_ckpt[k] = value  # 对于其他键，直接添加

        # 修改卷积层
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # new_ckpt['conv1.weight'] = state_dict['backbone.conv1.weight'].view(64, 3, 3, 3)
        # 加载状态字典
        model.load_state_dict(new_ckpt,strict=False)
        model.fc = nn.Identity()
        model.maxpool = nn.Identity()

        # 获取所有层的参数
        model_parameters = {name: param for name, param in model.named_parameters()}

        # 打印所有层的参数名称、形状和值
        # for name, param in model_parameters.items():
            # print(f"Layer: {name}, Shape: {param.shape}")
            # print(f"Parameters: {param.detach().numpy()}")  # 打印参数值

        return model




