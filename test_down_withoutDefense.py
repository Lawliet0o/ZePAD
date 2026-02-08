import torch
import os
import torch
import argparse
import random
import json
import numpy as np
from pathlib import Path
import argparse
from utils.predict import test, adv_test, fr_test, make_print_to_file
from utils.data_loder import load_data
from utils.load_model import load_encoder
from models.linear import NonLinearClassifier
def arg_parse():
    parser = argparse.ArgumentParser(description='Test the performance of helper_method')
    parser.add_argument('--seed', default=100, type=int, help='which seed the code runs on')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu the code runs on')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'stl10', 'gtsrb', 'imagenet','animals10','svhn'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--type', default='gan_per')
    parser.add_argument('--pre_dataset', default='cifar10', choices=['cifar10', 'imagenet'])
    parser.add_argument('--ssl_method', default='supcon', choices=['simclr', 'byol', 'barlow_twins', 'deepclusterv2', 'dino', 'mocov3', 'mocov2plus', 'nnclr', 'ressl', 'simsiam', 'supcon', 'swav', 'vibcreg', 'vicreg', 'wmse'])
    parser.add_argument('--sup_dataset', default='cifar10')
    parser.add_argument('--criterion', default='nce', choices=['cos', 'nt', 'nce'])
    parser.add_argument('--noise_percentage', type=float, default=0.03)
    parser.add_argument('--eps', type=int, default=10)
    args = parser.parse_args()
    return args

def classify(args, encoder, F, uap):
    data = args.dataset
    # log_save_path = path
    train_loader, test_loader = load_data(data, args.batch_size, Path('../dataset'))

    results = {'clean_acc_t1': [], 'adv_acc_t1': [], 'decline_t1': [], 'clean_acc_t5': [], 'adv_acc_t5': [],
               'decline_t5': [], 'attack_success_rate': []}

    F.cuda()
    encoder.cuda()
    encoder.eval()
    clean_acc_t1, clean_acc_t5 = test(args, encoder, F, test_loader, data)
    adv_acc_t1, adv_acc_t5 = adv_test(args, encoder, F, test_loader, uap, data)
    attack_success_rate = fr_test(args, encoder, F, test_loader, uap, data)
    decline_t1 = ((clean_acc_t1 - adv_acc_t1) / clean_acc_t1) * 100
    decline_t5 = ((clean_acc_t5 - adv_acc_t5) / clean_acc_t5) * 100
    results['clean_acc_t1'].append(clean_acc_t1)
    results['clean_acc_t5'].append(clean_acc_t5)
    results['decline_t1'].append(decline_t1)
    results['adv_acc_t1'].append(adv_acc_t1)
    results['adv_acc_t5'].append(adv_acc_t5)
    results['decline_t5'].append(decline_t5)
    results['attack_success_rate'].append(attack_success_rate)

    print('Top1 test acc: %.4f, Top1 Adv_test acc: %.4f, Fooling rate: %.4f'
        % (clean_acc_t1, adv_acc_t1, attack_success_rate))

    return clean_acc_t1, adv_acc_t1, decline_t1, attack_success_rate, clean_acc_t5, adv_acc_t5, decline_t5

def main():

    args = arg_parse()
    # Set the seed and determine the GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.set_printoptions(profile="full")
    torch.cuda.synchronize()

    args.eps = args.eps / 255
    # Logging
    log_save_path = os.path.join('./advencoder', str(args.pre_dataset), 'log', 'down_test', str(args.type), str(args.ssl_method), str(args.sup_dataset), str(args.dataset))
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)

    now_time = make_print_to_file(path=log_save_path)

    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)

    # Dump args
    with open(log_save_path + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)
    # load encoder
    encoder = load_encoder(args)

    # downstream task
    if args.dataset == 'imagenet':
        num_classes = 20
    elif args.dataset == 'gtsrb':
        num_classes = 43
    else:
        num_classes = 10

    classifier = torch.nn.DataParallel(NonLinearClassifier(feat_dim=512, num_classes=num_classes))

    # load classifier
    classifier_path = os.path.join('./clean_downstream', str(args.pre_dataset), str(args.ssl_method),str(args.dataset))
    if args.dataset == 'imagenet':
        encoder_path = [Path(classifier_path) / ckpt for ckpt in os.listdir(Path(classifier_path)) if ckpt.endswith("50.pth")][0]
    else:
        encoder_path = [Path(classifier_path) / ckpt for ckpt in os.listdir(Path(classifier_path)) if ckpt.endswith("20.pth")][0]

    checkpoint = torch.load(encoder_path)
    new_state_dict = {f"module.{k}": v for k, v in checkpoint.items()}
    classifier.load_state_dict(new_state_dict, strict=False)
    # classifier.load_state_dict(checkpoint)


    print('Day: %s, Target encoder:%s, Attack type: %s, Downstream task:%s'% (now_time, args.ssl_method, args.type, args.dataset))
    print("######################################  Clean Test Start ######################################")


    uap_load_path = os.path.join('./advencoder', str(args.pre_dataset), 'uap_results', str(args.type), str(args.ssl_method),
                                    str(args.sup_dataset), str(args.criterion), str(args.eps))
    uap_path = [Path(uap_load_path) / ckpt for ckpt in os.listdir(Path(uap_load_path)) if ckpt.endswith("20.pt")][0]




    UAP = torch.load(uap_path)

    clean_acc_t1, adv_acc_t1, decline_t1, attack_success_rate, clean_acc_t5, adv_acc_t5, decline_t5 = classify(args, encoder, classifier, UAP)
    print('Benign Accuracy: %.4f%%'% (clean_acc_t1))
    print('Adv Accuracy: %.4f%%'% (adv_acc_t1))
    print('Attack Success Rate: %.4f%%' % (attack_success_rate))

if __name__ == "__main__":
    main()
    


