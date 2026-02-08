import torch
import torch.nn as nn
from models.resnet import resnet18  
from models.pretrained import load_pretrained_model
import copy
class CustomEncoder(nn.Module):
    def __init__(self, args, num_classes=10):
        super(CustomEncoder, self).__init__()
        
        # 创建预训练模型的副本
        self.encoder = load_pretrained_model(args)
        # print(id(pretrained_model))
        print(id(self.encoder))
    

        # 处理最后一层的参数
        self.initialize_last_layer()

    def forward(self, x):
        # 前向传播函数，与预训练模型一致
        y = self.encoder(x)
        return y

    def initialize_last_layer(self):
        """
        随机初始化最后一层的参数。
        """
        param_names = list(self.encoder.state_dict().keys())
        print(param_names)
        # 假设最后一层是最后定义的卷积层或 BatchNorm 层
        # last_layer_name = "layer4.1.bn2.weight"
        last_layer_names = ["layer4.1.bn2.weight","layer4.1.bn2.bias","layer4.1.conv2.weight"]
        # last_layer_names = ["layer4.1.conv2.weight"]
        # last_layer_names = ["layer4.1.bn2.weight","layer4.1.bn2.bias"]
        # last_layer_names = ['layer4.0.conv1.weight','layer4.0.bn1.weight','layer4.0.bn1.bias','layer4.0.conv2.weight','layer4.0.bn2.weight','layer4.0.bn2.bias','layer4.1.conv1.weight','layer4.1.bn1.weight','layer4.1.bn1.bias',"layer4.1.bn2.weight","layer4.1.bn2.bias","layer4.1.conv2.weight"]
        state_dict = self.encoder.state_dict()
        for last_layer_name in last_layer_names:
            # 获取最后一层的权重和偏差
            last_layer_weight = state_dict[last_layer_name]
            # print(last_layer_name)
            #dropout
            # print(last_layer_weight)
            dropoutMask = torch.rand_like(last_layer_weight) > 0.4
            # print(dropoutMask)
            droppedWeigth = last_layer_weight * dropoutMask
            # print(droppedWeigth)
            # 噪声
            noise = torch.randn_like(last_layer_weight) * 0.01
            # print(noise)
            finalWeight = noise + droppedWeigth
            # print(finalWeight)
            state_dict[last_layer_name] = finalWeight
            # state_dict[last_layer_name] = torch.randn_like(state_dict[last_layer_name])

        self.encoder.load_state_dict(state_dict)
