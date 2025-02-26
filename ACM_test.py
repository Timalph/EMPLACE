#import
#import rectangular vit
from utils import CityPulsePano
from utils import create_dir_name
import os
from csv import writer
import argparse
import torch
import numpy as np
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
import pickle
import pandas as pd
import random
from glob import glob
import generate_triplets as gt
from Rectangular_VIT import VisionTransformer
import random
from torchvision.utils import save_image
from adjusted_factory import create_vit
import sys
from utils import calculate_metrics
from transformers import ViTImageProcessor, ViTModel
from utils import perform_forward_2
from transformers import CLIPProcessor, CLIPModel
#from code_dinodino.build_dinodino import build_dinodino
### Use same optimizer and clipping as Citypulse

def latexify(n):
    return str(np.round(n, 3))[1:]

def main(args):
    seed=args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if not torch.cuda.is_available():
        sys.exit(0)
    if args.ATM_buildings:
        args.testfile = 'ATM_buildings_testset.csv'
    if args.ATM_buildings_square:
        normalized_splits = np.array(args.splits)/100
        splits = '{}train_{}_val_{}_test_{}'.format('{}_'.format(args.balanced) if args.balanced else '',normalized_splits[0], normalized_splits[1], normalized_splits[2])

        args.testfile  =  'ATM_square_trainsplits/ATM_buildings_square_{}_{}_testset.csv'.format(args.ATM_buildings_square, splits)
        #args.testfile = 'ATM_buildings_square_{}_testset.csv'.format(args.ATM_buildings_square)
    print(args.testfile)
    ### DATASET
    mean, sd = pickle.load(open('./means/mean_sd_{}.pickle'.format(args.amount_of_clusters),'rb'))
    testset= CityPulsePano(args.testfile,args.citypulse_root,  mean, sd, device=device, nojitter=args.nojitter, softmax=args.softmax)
    testlabels = np.array(pd.read_csv(args.testfile)['change_flag']).astype(int)
    print('Testset of size: {}'.format(testset.__len__()))
    test_dataloader = DataLoader(testset, batch_size=1)
    ### MODEL

    if args.headtype == 'DINOV2':
        model = ViTModel.from_pretrained('facebook/dino-vitb16')
    #elif args.headtype == 'DINODINO':
    #    model = build_dinodino()
    elif args.headtype=='resnet':
        model = torch.hub.load("pytorch/vision:v0.13.0", "resnet50", weights="IMAGENET1K_V2")
    elif args.headtype=='clip':
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    elif args.headtype == 'DINODINO14' or args.headtype == 'DINODINO':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    else:
        model = VisionTransformer(image_size=args.image_size, patch_size=args.patch_size, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072)
        weight_url = 'https://download.pytorch.org/models/vit_b_16-c867db91.pth'

    ############################################################ LOAD PRETRAINED ##################################################################
    print('Loading pretrained {}'.format(args.dirpath))
    checkpoint_dict = torch.load(args.dirpath + 'best_epoch.pth')
    unparalleled_checkpoint_dict = {}
    try:
        model.load_state_dict(checkpoint_dict['model_state_dict'])
    except KeyError:
        for k, v in checkpoint_dict['model_state_dict'].items():
            unparalleled_checkpoint_dict[k[7:]] = v
        model.load_state_dict(unparalleled_checkpoint_dict)
    ##############################################################################################################################################
    # eigenlijk 768
    #### We have input_dim, headtype   

    #write create_model in utils
    #model = create_model(input_dim, model, headtype)
    if args.headtype == 'cls' or args.headtype == 'DINOV2' or args.headtype=='DINODINO' or args.headtype=='DINODINO14':
        input_dim = 768
    elif args.headtype == 'attention' or args.headtype=='resnet':
        input_dim = 1000
    elif args.headtype == 'clip':
        input_dim = 512
    else:
        raise NotImplementedError('Only \'cls\', \'DINOV2\', \'DINODINO\' or \'attention\' are valid options')
    
    model.to(device)
    model.eval()

    if not args.softmax:
        if args.cls_style:
            head = torch.nn.Linear(1 + (input_dim*3), 1, bias=True if args.headbias else False)            

        elif args.cls_only:
            head = torch.nn.Linear(1, 1, bias=True if args.headbias else False)
        else:
            head = torch.nn.Linear(input_dim*3, 1, bias=True if args.headbias else False)

        sigmoid_or_softmax = torch.nn.Sigmoid()
    elif args.softmax:
        head = torch.nn.Linear(input_dim*3, 2, bias=True if args.headbias else False)
        sigmoid_or_softmax = torch.nn.Softmax(dim=1)
    if args.triple_head:
        head = torch.nn.Sequential(torch.nn.Linear(input_dim*3, input_dim, bias=True if args.headbias else False),
                                   torch.nn.ReLU(),
                                   torch.nn.Linear(input_dim, 256, bias=True if args.headbias else False),
                                   torch.nn.ReLU(),
                                   torch.nn.Linear(256, 2, bias=True if args.headbias else False))
    head.load_state_dict(checkpoint_dict['head'])
    head = head.to(device)
    
    head.eval()
    pdist = nn.PairwiseDistance(p=2)
    ##############################################################################################################################################################
    test_accs = []
    test_precisions = []
    test_recalls = []
    test_f1s = []
    all_preds = []
    start_epoch_time = time()
    with torch.no_grad():
        for idx, (imgs, label) in enumerate(test_dataloader):
            im1, im2  = imgs

            out1, out2 = perform_forward_2(model, im1, im2, args.headtype)
            if args.cls_style:
                x = torch.concat([out1, out2, out2-out1, pdist(out1, out2).unsqueeze(1)], dim=1) 
            if args.cls_only:
                    x = pdist(out1, out2).unsqueeze(1)     
            else:
                x = torch.concat([out1, out2, out2-out1], dim=1)   

            pred = head(x)
            pred = sigmoid_or_softmax(pred)

            if args.softmax:
                acc_preds = torch.argmax(pred, dim=1).float()
            else:   
                acc_preds = (pred.view(-1)>0.5).float()

            acc = torch.sum(torch.where(acc_preds==label, 1, 0))/len(acc_preds)
            acc  = float(acc.detach().cpu().numpy())
            print('Acc:', np.round(np.mean(test_accs),4))
            print('Prec:', np.round(np.mean(test_precisions),4))
            print('Rec:', np.round(np.mean(test_recalls),4))
            print('F1:', np.round(np.mean(test_f1s),4))
            print('{}/{}'.format(idx, test_dataloader.__len__()))  
            test_accs.append(acc)
            all_preds.append(int(acc_preds.detach().cpu().numpy()))
            precision, recall, f1 = calculate_metrics(testlabels[:idx+1], np.array(all_preds))
            test_precisions.append(precision)
            test_recalls.append(recall)
            test_f1s.append(f1)

        print(time() - start_epoch_time)
        print('test_acc:', np.round(np.mean(test_accs),4))
        print('test_precision', np.round(np.mean(test_precisions),4))
        print('test_recall', np.round(np.mean(test_recalls),4))
        print('test_f1', np.round(np.mean(test_f1s),4))
        #### Eval
                
        with open(args.dirpath + 'test.txt', 'a') as f:
            f.write('Testing epoch {} took {} and got the following results:\naccuracy: {:0.8f}\nprecision: {:0.8f}recall: {:0.8f}F1: {:0.8f}'.format(checkpoint_dict['epoch'],
             time() - start_epoch_time, np.mean(test_accs), np.mean(test_precisions), np.mean(test_recalls), np.mean(test_f1s)))
            f.write('\n')
        with open(args.dirpath+'test_results.pickle', 'wb') as pfile:
            pickle.dump(all_preds, pfile)

        if args.ATM_buildings:
            with open('ATM_buildings_tests.csv', 'a') as obj:
                writer_obj = writer(obj)
                writer_obj.writerow([args.dirpath, np.mean(test_accs), np.mean(test_precisions), np.mean(test_recalls), np.mean(test_f1s)])
        #imgtype, dim ,batch, ppx, pos, freeze, acc, prec, rec, f1
        try:
            batch_size = args.dirpath.split('batch_')[1].split('_')[0]
            imgs1 = args.dirpath.split('img_')[1].split('_')[0]
            imgs2 = args.dirpath.split('img_')[1].split('_')[1]
            if imgs1 == imgs2 == '224':
                ppx, pos, neg = ['---','---','---']
            else:
                ppx = args.dirpath.split('ppx')[1].split('_')[0]
                pos = args.dirpath.split('ppx')[1].split('_')[1][1:]
                neg = args.dirpath.split('ppx')[1].split('_')[2][1:]
        except IndexError:
            batch_size='16'
            imgs1='210'
            imgs2='700'
            ppx, pos, neg = ['---','---','---']
        freeze = 'cmark' if 'nofreeze' in args.dirpath else 'xmark'
        split_string = '{}/{}/{}'.format(str(args.splits[0]), args.splits[1], args.splits[2])
        if args.ATM_buildings_square:
            with open('novdinopre_nov_ATM_buildings_square_tests.csv', 'a') as obj:
                writer_obj = writer(obj)
                writer_obj.writerow([args.dirpath, np.mean(test_accs), np.mean(test_precisions), np.mean(test_recalls), np.mean(test_f1s)])

            with open('dinopre_nov_{}_latex_ATM_buildings_square{}{}{}{}_tests.txt'.format(args.headtype, args.ATM_buildings_square.split('_')[1] if 'boom' in args.ATM_buildings_square else '','_{}'.format(args.balanced) if args.balanced else '', '_cls_only' if args.cls_only else '', '_{}'.format(args.seed if args.seed else '')), 'a') as obj:
                latex_line = ''.join(x+'&' for x in [args.ATM_buildings_square[:2], '{}x{}'.format(imgs1, imgs2), batch_size, ppx, pos, neg, split_string, freeze, 
                latexify(np.mean(test_accs)), latexify(np.mean(test_precisions)), latexify(np.mean(test_recalls)), latexify(np.mean(test_f1s))])
                print(latex_line)
                obj.write(latex_line[:-1] + "\\\\")
                obj.write('\n')
        else:
            with open(args.resultfile, 'a') as obj:
                writer_obj = writer(obj)
                writer_obj.writerow([args.dirpath, np.mean(test_accs), np.mean(test_precisions), np.mean(test_recalls), np.mean(test_f1s)])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirpath', type=str, default='./')
    parser.add_argument('--testfile', type=str, default='ACM_testset_clusters_original.csv')
    parser.add_argument('--image_size', nargs='+', type=int, default=[128,400])
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--amount_of_clusters', type=int, default=10000)
    parser.add_argument('--nojitter', type=int, default=1)
    parser.add_argument('--headtype', type=str, default='attention')
    parser.add_argument('--headbias', type=int, default=1)
    parser.add_argument('--softmax', type=int, default=0)
    parser.add_argument('--citypulse_root', type=str, default = '../../citypulse/')
    parser.add_argument('--resultfile', type=str, default = 'citypulse_tests.csv')
    parser.add_argument('--triple_head', type=int, default=0)
    parser.add_argument('--cls_style', type=int, default=0)
    parser.add_argument('--cls_only', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--splits', nargs='+', type=int)
    parser.add_argument('--ATM_buildings', type=int, default=0)
    parser.add_argument('--ATM_buildings_square', type=str, default='')
    parser.add_argument('--balanced', type=str, default='')# balanced or distractors
    args = parser.parse_args()
    
    main(args)


#def func
#split args


#create dataloader

#load in weights

#create model


#train on data




#test data


# Train rect vit on half split, then test on half split - get base acc

# add linear layer to pretrained unsup, train on half split, test on half split




#Ablation - train with and without jitter
#train with and wihout hflip




