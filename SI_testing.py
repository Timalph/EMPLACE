import numpy as np
import os
import torch
import torchvision
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
import torch.optim as optim
from time import sleep
from time import time
import argparse
import pickle
import pandas as pd
import random
from glob import glob
import generate_triplets as gt
from Rectangular_VIT import VisionTransformer
import random
from torchvision.utils import save_image
from adjusted_factory import create_vit
from utils import cut_and_flip
from utils import calc_mean_and_sd
from utils import sum_params
from utils import Tweaked_TripletMarginLoss
from utils import Tweaked_TripletMarginLoss_Adaptive
from utils import Tweaked_TripletMarginLoss_Linear
from utils import create_dir_name_ATM
from utils import Change_AMS
import sys
from transformers import ViTImageProcessor, ViTModel
from utils import perform_forward_3
def imgtotensor(i, mean, sd):
        transform =   transforms.Compose([
            #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, sd)])
        t = transform(i)
        return t

def cut_and_flip(x, randint):
    
    left = x[:,:,:randint]
    right = x[:,:,randint:]
    
    return torch.cat((right,left), dim=2)

def Tweaked_TripletMarginLoss(a,p,n, no_margin=False):
    pdist = nn.PairwiseDistance(p=2)    
    pos_distance = pdist(a,p)
    neg_distance = pdist(a,n)
    if not no_margin:
        diff = pos_distance - neg_distance +1
    else:
        diff = pos_distance - neg_distance


    max_0 = torch.where(diff>0, diff, 0)
    loss = torch.mean(max_0)
    acc = sum(max_0==0)/len(max_0)
    return loss, acc



def SI_test(ppx, pos, neg, npx, model, csv_file_test='AC-1M_test.csv'):

    mean, sd = pickle.load(open('./means/mean_sd_{}.pickle'.format(10000),'rb'))
    print('filtering for tiny..')
    df = pd.read_csv(csv_file_test)
    
    tinyfile = 'ATM_testsets/{}_{}_{}_{}_tcg.csv'.format(ppx, pos, neg, npx)

    try:
        df = pd.read_csv(tinyfile)
    except FileNotFoundError:
        csv_file_test_tiny = gt.filter_distances_after_saving(df, pos, neg, pos_prox = ppx, neg_prox = npx, step=250000)
        csv_file_test_tiny.to_csv(tinyfile)

    if 'img_126_420' in modelpath:
        input_dir = '/home/talpher/phd/sscd/antenna_420_126_test'
    elif 'img_210_700' in modelpath:
        input_dir = '/home/talpher/antenna_700_210_test'
    else:
        input_dir = '/home/talpher/antenna_700_210_test'
 
    print(input_dir)
    dataset_tiny = Change_AMS(tinyfile, input_dir, mean, sd, nojitter=True)
    ## GPU
    test_dataloader_tiny = DataLoader(dataset_tiny, batch_size=1)
    
    tiny_test_accs = []
    start = time()
    ### CALCULATING ACC
    for idx, (triplet_imgs, triplet_labels) in enumerate(test_dataloader_tiny):

        anc, pos, neg = triplet_imgs
        
        anc = anc.to(device)
        pos = pos.to(device)
        neg = neg.to(device)

        anchor_out, positive_out, negative_out = perform_forward_3(model, anc, pos, neg, architecture)

        # anchor_out = model.forward(anc, interpolate_pos_encoding=True)[1]
        # positive_out = model.forward(pos, interpolate_pos_encoding=True)[1]
        # negative_out = model.forward(neg, interpolate_pos_encoding=True)[1]

        _, acc = Tweaked_TripletMarginLoss(anchor_out, positive_out, negative_out, no_margin=True)
        acc  = float(acc.detach().cpu().numpy())
        print('Acc:', acc)
        print('{}/{}'.format(idx, test_dataloader_tiny.__len__()))  
        tiny_test_accs.append(acc)

    print('tiny acc:', np.mean(tiny_test_accs))
    return np.mean(tiny_test_accs)


csvname = 'SI_test_results_all_balanced.csv'
try:
    df = pd.read_csv(csvname)

except FileNotFoundError:
    df = pd.DataFrame(columns = ['modelpath', 'SI_1', 'SI_2', 'SI_3', 'SI_4', 'B_ACC'])

for i, d in enumerate(['img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx1_p31_n375_gc0.5_DINODINO14_linear_loss',
                        'img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n750_gc0.5_DINODINO14_linear_loss',
                        'img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n1125_gc0.5_DINODINO14_linear_loss',
                        'img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n1500_gc0.5_DINODINO14_linear_loss',
                        
                        'img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx1_p31_n375_gc0.5_DINODINO14_fixed_margin_loss',
                        'img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n750_gc0.5_DINODINO14_fixed_margin_loss',
                        'img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n1125_gc0.5_DINODINO14_fixed_margin_loss',
                        'img_210_700_batch_64_patch_14_nojitter_tcg/attention_ppx275_p475_n1500_gc0.5_DINODINO14_fixed_margin_loss',

                        'img_210_700_batch_64_patch_14_nojitter_tcg_nocnf/attention_ppx1_p31_n375_gc0.5_DINODINO14_linear_loss',
                        'img_210_700_batch_64_patch_14_nojitter_tcg_nocnf/attention_ppx275_p475_n750_gc0.5_DINODINO14_linear_loss',
                        'img_210_700_batch_64_patch_14_nojitter_tcg_nocnf/attention_ppx275_p475_n1125_gc0.5_DINODINO14_linear_loss',
                        'img_210_700_batch_64_patch_14_nojitter_tcg_nocnf/attention_ppx275_p475_n1500_gc0.5_DINODINO14_linear_loss']:

    d = os.path.join('DINODINO14_ATM_saved_models', d)
#    d = os.path.join('DINODINO14_ATM_continuous_saved_models', d)

    ############################################ LOADING MODEL
    
    print('Loading checkpoint...')
    modelpath = d
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    architecture = 'DINODINO14'
    if d!='vanilla':
        try:
            checkpoint_dict = torch.load(os.path.join(modelpath, 'best_epoch.pth'))
            model.load_state_dict(checkpoint_dict['model_state_dict'])
        except RuntimeError:
            unparalleled_checkpoint_dict = {}
            for k, v in checkpoint_dict['model_state_dict'].items():
                print(k)
                unparalleled_checkpoint_dict[k[7:]] = v
            model.load_state_dict(unparalleled_checkpoint_dict)
        print('Loaded model!')
    else:
        print('Running Vanilla!')
        pass


    if not torch.cuda.is_available():
        print('No GPU')
        sys.exit(0) 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    ############################################
    
    acc = [d]
    ### ADD NPX, OTHERWISE SI_1 IS THE SUPERSET OF WHAT COMES AFTER
    for SI in [(1, 31, 375, 9999), # 827
               (275, 475, 750, 1125), # 2578    
               (275, 475, 1125, 1500), # 1679
               (275, 475, 1500, 9999)]: # 1081
    #           (120, 240, 750, 9999), #1411
    
        ppx, pos, neg, npx = SI


        acc.append(SI_test(ppx, pos,neg, npx, model))
    balance_array = np.array([827, 2578, 1679, 1081])
    B_ACC = np.sum(np.array(acc[1:]) * balance_array)/np.sum(balance_array)
    acc.append(B_ACC)
    df.loc[len(df)] = acc

    df.to_csv(csvname, index=False)



