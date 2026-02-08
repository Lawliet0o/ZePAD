import os
import torch
from pathlib import Path
import torch.nn as nn
from models.resnet import resnet18
from models.linear import NonLinearClassifier
import timm
# victim model source zoo: https://github.com/vturrisi/solo-learn

def load_encoder(args):

    if args.pre_dataset == 'cifar10':
        num_classes = 10
        encoder_folder = os.path.join('./pretrained_encoders', 'cifar10', str(args.ssl_method))
        encoder_path = [Path(encoder_folder) / ckpt for ckpt in os.listdir(Path(encoder_folder)) if ckpt.endswith(".ckpt")][0]
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
                new_ckpt[k] = value 
        if True:
            model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                                          bias=False)

        model.load_state_dict(new_ckpt,strict=False)
        model.fc = nn.Identity()
        model.maxpool = nn.Identity()
    elif args.pre_dataset == 'imagenet':
        num_classes = 100
        encoder_folder = os.path.join('./pretrained_encoders', 'imagenet', str(args.ssl_method))
        encoder_path = [Path(encoder_folder) / ckpt for ckpt in os.listdir(Path(encoder_folder)) if ckpt.endswith(".ckpt")][0]
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
                new_ckpt[k] = value 
        if True:
            model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=1,
                                          bias=False)

        model.load_state_dict(new_ckpt,strict=False)
        model.fc = nn.Identity()
        model.maxpool = nn.Identity()

    return model


def load_finetuned_encoder_F(args):

    encoder_path = os.path.join('./finetuned', str(args.pre_dataset), str(args.ssl_method), str(args.ft_dataset), 
                                'encoder')
    checkpoint = [Path(encoder_path) / ckpt for ckpt in os.listdir(Path(encoder_path)) if ckpt.endswith("last.pth")][0]
    encoder = torch.load(checkpoint)

    f_path = os.path.join('./finetuned', str(args.pre_dataset), str(args.ssl_method), str(args.ft_dataset), 'f')
    f_checkpoint = [Path(f_path) / ckpt for ckpt in os.listdir(Path(f_path)) if ckpt.endswith("last.pth")][0]

    F = torch.load(f_checkpoint)
    return encoder,F

def load_dual_classfier(args):
    if args.dataset == 'gtsrb':
        num_classes = 43
    elif args.dataset == 'imagenet':
        num_classes = 20
    else:
        num_classes = 10
    F1 = NonLinearClassifier(feat_dim=512, num_classes=num_classes)
    F2 = NonLinearClassifier(feat_dim=512, num_classes=num_classes)

    F1_path = os.path.join('./aft_downstream_se', str(args.pre_dataset), str(args.ssl_method), str(args.ft_dataset),str(args.dataset))
    if args.dataset == 'imagenet':
        F1_checkpoint = [Path(F1_path) / ckpt for ckpt in os.listdir(Path(F1_path)) if ckpt.endswith("50.pth")][0]
    else:
        F1_checkpoint = [Path(F1_path) / ckpt for ckpt in os.listdir(Path(F1_path)) if ckpt.endswith("20.pth")][0]
    checkpoint1 = torch.load(F1_checkpoint)
    F1_dict = {f"{k}":v for k, v in checkpoint1.items()}
    F1.load_state_dict(F1_dict)
    
    F2_path = os.path.join('./clean_downstream', str(args.pre_dataset), str(args.helper), str(args.dataset))

    if os.path.exists(F2_path):
        if args.dataset == 'imagenet':
            F2_checkpoint = [Path(F2_path) / ckpt for ckpt in os.listdir(Path(F2_path)) if ckpt.endswith("50.pth")][0]
        else:
            F2_checkpoint = [Path(F2_path) / ckpt for ckpt in os.listdir(Path(F2_path)) if ckpt.endswith("20.pth")][0]
    else:
        F2_checkpoint = None
    # F2_checkpoint = [Path(F2_path) / ckpt for ckpt in os.listdir(Path(F2_path)) if ckpt.endswith("20.pth")][0]
    if F2_checkpoint != None:
        checkpoint2 = torch.load(F2_checkpoint)
        F2_dict = {f"{k}":v for k, v in checkpoint2.items()}
        F2.load_state_dict(F2_dict)
    else:
        F2 = None
    return F1,F2



