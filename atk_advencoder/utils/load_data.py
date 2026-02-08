import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from pathlib import Path

def load_data(data, bs, data_dir=Path('../dataset')):

    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    # transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    if data == 'cifar10':
        train_dataset = datasets.CIFAR10(data_dir / 'cifar10', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(data_dir / 'cifar10', train=False, download=True, transform=transform)

    elif data == 'stl10':
        train_dataset = datasets.STL10(data_dir / 'stl10', split="train", download=True, transform=transform)
        test_dataset = datasets.STL10(data_dir / 'stl10', split="test", download=True, transform=transform)

    elif data == 'gtsrb':
        train_dataset = datasets.GTSRB(data_dir / 'GTSRB', split="train", download=False, transform = transform)
        test_dataset = datasets.GTSRB(data_dir / 'GTSRB', split="test", download=False, transform=transform)
    elif data == 'imagenet':
        train_dataset = datasets.ImageFolder(data_dir / 'imagenet20/train', transform = transform)
        test_dataset = datasets.ImageFolder(data_dir / 'imagenet20/val', transform = transform)
    elif data == 'animals10':
        train_dataset = datasets.ImageFolder(data_dir / 'animals10/raw-img', transform=transform)
        train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [20000, len(train_dataset)-20000])
    elif data == 'svhn':
        train_dataset = datasets.SVHN(data_dir / 'SVHN', split='train', download=True, transform=transform)
        test_dataset = datasets.SVHN(data_dir / 'SVHN', split='test', download=True, transform=transform)
    print(f'Train dataset: {len(train_dataset)}, Test dataset: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=5, drop_last=False, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=bs, num_workers=5, drop_last=False, shuffle=False)

    return train_loader, test_loader



def normalzie(args, x):

    if args.dataset == 'cifar10':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        normalizer = transforms.Normalize(mean=mean, std=std)
        return normalizer(x)

    elif args.dataset == 'stl10':
        mean = (0.44087798, 0.42790666, 0.38678814)
        std = (0.25507198, 0.24801506, 0.25641308)
        normalizer = transforms.Normalize(mean=mean, std=std)
        return normalizer(x)

    elif args.dataset == 'gtsrb':
        mean = (0.44087798, 0.42790666, 0.38678814)
        std = (0.25507198, 0.24801506, 0.25641308)
        normalizer = transforms.Normalize(mean=mean, std=std)
        return normalizer(x)

    else:
        return x
    
    




