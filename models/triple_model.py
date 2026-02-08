import torch.nn as nn
from utils.load_model import load_encoder,load_dual_classfier,load_finetuned_encoder_F
from utils.data_loder import  normalzie
from utils.confidence import federal_with_zepad,federal_with_equal,federal_with_entropy
# import numpy as np
import torch
import torch.nn.functional as F
import os

class triple_model(nn.Module):
    def __init__(self, args):
        super(triple_model, self).__init__()
        self.args = args
        # victim
        self.victim , _ = load_finetuned_encoder_F(args) 
        self.F1, _  = load_dual_classfier(args)

        # helper1
        ssl_method = args.ssl_method
        pre_dataset = args.pre_dataset
        args.ssl_method = args.helper
        args.pre_dataset = 'cifar10'
        self.helper1 = load_encoder(args)
        args.ssl_method = ssl_method
        _ , self.F2 = load_dual_classfier(args)
        # helper2
        args.ssl_method = 'byol'
        # args.pre_dataset = helper_pre_dataset
        args.pre_dataset = 'imagenet'
        # args.pre_dataset = 'cifar10'
        self.helper2 , _ = load_finetuned_encoder_F(args)
        self.F3 , _ = load_dual_classfier(args)
        args.ssl_method = ssl_method
        args.pre_dataset = pre_dataset

    
    def forward(self, x):
        vic_feature = self.victim(x)
        help1_feature = self.helper1(x)
        help2_feature = self.helper2(x)

        victimOutput = self.F1(vic_feature)
        help1Output = self.F2(help1_feature)
        help2Output = self.F3(help2_feature)
        if self.args.confidence == "zepad":
            return federal_with_zepad(victimOutput,help1Output,help2Output)
        elif self.args.confidence == "equal":
            return federal_with_equal(victimOutput,help1Output,help2Output)
        elif self.args.confidence == "entropy":
            return federal_with_entropy(victimOutput,help1Output,help2Output)
