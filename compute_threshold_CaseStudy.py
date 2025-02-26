import pandas as pd
import sys
import os
from utils import CityPulsePano
#from utils import create_dir_name
import os
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
from utils import sum_params
import gc
#import generate_triplets as gt
#from Rectangular_VIT import VisionTransformer
import random
from torchvision.utils import save_image
#from adjusted_factory import create_vit
import sys
from sklearn.metrics.pairwise import paired_distances
from utils import calculate_metrics
#from utils import cut_and_flip, copy2zfs_mp
#from utils import location_sampling
#from utils import perform_forward_2
#from transformers import ViTImageProcessor, ViTModel
#from transformers import CLIPProcessor, CLIPModel

# def load_model(args):
#     model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
#     if not args.pretrained:
#         print('DINODINO14 vanilla loaded!')
#         return model
#     checkpoint_dict = torch.load(args.pretrained)
#     if args.vanilla_posembed:
#         try:
#             checkpoint_dict['model_state_dict'].pop('pos_embed')
#         except:
#             checkpoint_dict['model_state_dict'].pop('module.pos_embed')

#         print('RUNNING WITH VANILLA POSITIONAL EMBEDDING')
#     try:
#         model.load_state_dict(checkpoint_dict['model_state_dict'], strict=not args.vanilla_posembed)
#     except RuntimeError:
#         unparalleled_checkpoint_dict = {}
#         for k, v in checkpoint_dict['model_state_dict'].items():
#             print(k)
#             unparalleled_checkpoint_dict[k[7:]] = v
#         model.load_state_dict(unparalleled_checkpoint_dict, strict=not args.vanilla_posembed)
#         print('DINODINO14 pretrained weights loaded!')
#     return model


def load_model(args):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    if args.pretrained == 'vanilla':
        print('DINODINO14 vanilla loaded!')
        return model
    checkpoint_dict = torch.load(args.pretrained)
    print('before loading weights: {}'.format(sum_params(model)))
    if args.vanilla_posembed:
        try:
            checkpoint_dict['model_state_dict'].pop('pos_embed')
        except:
            checkpoint_dict['model_state_dict'].pop('module.pos_embed')

        print('RUNNING WITH VANILLA POSITIONAL EMBEDDING')
    if 'module.norm.bias' in checkpoint_dict['model_state_dict'].keys():
        unparalleled_checkpoint_dict = {}
        for k, v in checkpoint_dict['model_state_dict'].items():
            unparalleled_checkpoint_dict[k[7:]] = v
        model.load_state_dict(unparalleled_checkpoint_dict, strict=not args.vanilla_posembed)
        print('DINODINO14 pretrained weights loaded using multigpu!')
    else:
        model.load_state_dict(checkpoint_dict['model_state_dict'], strict=not args.vanilla_posembed)
        print('after loading weights: {}'.format(sum_params(model)))
    return model

def latexify(acc, prec, rec, f1):
    vals = [str(np.round(x,3))[1:] for x in [acc, prec, rec, f1]]
    return "&${}\pm{}$&${}\pm{}$&${}\pm{}$&${}\pm{}$\\\\".format(vals[0], 0, vals[1], 0, vals[2], 0, vals[3], 0)


#img1 img2 img3 img4 0010

# try: 
#     df = pd.read_csv('boom_threshold_data.csv')
# except FileNotFoundError:
#     boom_df = pd.read_csv('ATM_square_boom.csv')
#     print(len(boom_df['seq_index'].unique()))
#     count = 0
#     df = pd.DataFrame(columns = ['img1', 'img2', 'img3', 'change_flag', 'seq_index', 'i'])
#     for seq_index_obj in boom_df.groupby('seq_index'):
#         seq_index, sub_df = seq_index_obj
#         if len(sub_df) < 3:
#             continue
#         for i in range(len(sub_df)-3):
#             consistent_change_candidate  = sub_df.iloc[i:i+3]['change_flag']
#             if list(consistent_change_candidate) == [0,0,1] or list(consistent_change_candidate) == [0,1,0] or list(consistent_change_candidate) == [0,0,0]: 
#                 #print(consistent_change_candidate)
#                 print(sub_df.iloc[i:i+3])
#                 count +=1
#                 df.loc[len(df)] = [list(sub_df['filename'])[0],
#                                     list(sub_df['filename'])[1],
#                                     list(sub_df['filename'])[2],
#                                     ''.join(str(x) for x in list(consistent_change_candidate)), 
#                                     sub_df.iloc[0]['seq_index'], i]
#                 #sys.exit(0)

#     print(count)
#     df.to_csv('boom_threshold_data.csv', index=False)
#         #print(sub_df['change_flag'])

def main(args):

    seed=args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    window_dim = args.window_dim
## Load model
## Model to device

## load mean and sd
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        sys.exit(0)

    model = load_model(args).to(device)
    mean, sd = pickle.load(open('./means/mean_sd_{}.pickle'.format(10000),'rb'))
    dataset_df = pd.read_csv('ATM_square_trainsplits/ATM_buildings_square_equirectangular_{}{}_train_0.7_val_0.2_test_0.1_{}set.csv'.format(args.object, seed if seed else '', args.split))
    # val_df = pd.read_csv('ATM_square_trainsplits/ATM_buildings_square_equirectangular_{}_train_0.7_val_0.2_test_0.1_valset.csv'.format(args.object))    
    # test_df = pd.read_csv('ATM_square_trainsplits/ATM_buildings_square_equirectangular_{}_train_0.7_val_0.2_test_0.1_testset.csv'.format(args.object))    
    
    dataset = CityPulsePano(dataset_df, args.citypulse_root, mean, sd, device=device, nojitter=True, softmax=False)
    # valset = CityPulsePano(val_df, args.citypulse_root, mean, sd, device=device, nojitter=True, softmax=False)
    # testset = CityPulsePano(test_df, args.citypulse_root, mean, sd, device=device, nojitter=True, softmax=False)

    print('Using split: {} of size: {}'.format(args.split, len(dataset)))
    selected_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, worker_init_fn=seed)
    # val_dataloader = DataLoader(valset,batch_size=1)




    ## Thresholds can be gridsearches linearly because this is a linear problem 
    ## FIX THIS VV
                #precision, recall, f1 = calculate_metrics(testlabels[:idx+1], np.array(all_preds))
                 #           acc = torch.sum(torch.where(acc_preds==label, 1, 0))/len(acc_preds)
    csv_name = os.path.join(args.pretrained.replace('best_epoch.pth', ''), '{}zeroshot_thresholdscores_{}{}.csv'.format('roc_' if args.roc else '',
        args.split, '_boom' if args.object == 'boom' else ''))
    if not args.roc:
        try:
            score_df = pd.read_csv(csv_name)
            if not 'seed' in score_df.columns:
                score_df.insert(loc=2, column='seed', value=0) 
                score_df.to_csv(csv_name, index=False)
            if args.relatexify:
                latexlines = []
                for i in range(len(score_df)):
                    latexlines.append(latexify(score_df.iloc[i]['acc'], 
                                            score_df.iloc[i]['prec'], 
                                            score_df.iloc[i]['rec'], 
                                            score_df.iloc[i]['f1']))
                score_df['latexline'] = latexlines

        except FileNotFoundError:
            score_df = pd.DataFrame(columns=['window_dim', 'threshold', 'seed','acc', 'prec', 'rec', 'f1', 'latexline'])
    else:
        score_df = pd.DataFrame(columns=['window_dim', 'threshold', 'seed','acc', 'prec', 'rec', 'f1', 'fpr', 'latexline'])


    token_distances_dict_filename = os.path.join(args.pretrained.replace('best_epoch.pth', ''),'token_distances_dict_{}_{}{}'.format(args.split, 'buildings' if args.object == 'balanced' else 'boom', seed if seed else ''))
    adaptive_threshold_dict_filename = os.path.join(args.pretrained.replace('best_epoch.pth', ''),'adaptive_threshold_dict_{}_{}{}.pickle'.format(args.split, 'buildings' if args.object == 'balanced' else 'boom', seed if seed else ''))

    try:
        token_distances_dict = pickle.load(open(token_distances_dict_filename, 'rb'))
        tokens_from_disk = True
    except FileNotFoundError:
        token_distances_dict = {}
        tokens_from_disk = False

    adaptive_threshold = args.adaptive_threshold
    if adaptive_threshold:
        try:
            adaptive_threshold_dict = pickle.load(open(adaptive_threshold_dict_filename, 'rb'))
        except FileNotFoundError:
            #Build it
            adaptive_threshold_dict = {}
            with torch.no_grad():
                for idx, (imgs, label) in enumerate(selected_dataloader):
                    print('Building adaptive_threshold_dict {}/{}'.format(idx, len(selected_dataloader)))
                    #print(dataset.__getdetails__(idx))
                    #sys.exit(0)
                    im1, im2 = imgs
                    ### Rewrite this!
                    if tokens_from_disk:
                        token_distances = token_distances_dict[idx]
                    else:
                        im1 = model.forward_features(im1)['x_norm_patchtokens'].squeeze(0)
                        im2 = model.forward_features(im2)['x_norm_patchtokens'].squeeze(0)

                    #cd, wi = calc_distances(im1, im2, window_size=[8,8], method='paired')
                    
                        token_distances = paired_distances(im1.cpu(), im2.cpu()).reshape(16, 16)
                        token_distances_dict[idx] = token_distances
                    try:
                        adaptive_threshold_dict[dataset.__getdetails__(idx)[0]].append(np.mean(token_distances))
                    except KeyError:
                        adaptive_threshold_dict[dataset.__getdetails__(idx)[0]] = [np.mean(token_distances)]
                for key, val in adaptive_threshold_dict.items():
                    adaptive_threshold_dict[key] = np.mean(adaptive_threshold_dict[key])
                with open(adaptive_threshold_dict_filename, 'wb') as obj:
                    pickle.dump(adaptive_threshold_dict, obj)
                print('saved to {}'.format(adaptive_threshold_dict_filename))
    

    # for threshold in multiple_thresholds:
    threshold = args.threshold
    ###MULTIPLE

    if args.multiple_threshold[0]:
        threshold_range_low, threshold_range_high = args.multiple_threshold
        threshold_list = list(range(threshold_range_low, threshold_range_high+1))
    else:
        threshold_list = [threshold]

    for threshold in threshold_list:
        gt_labels = []
        pred_labels = []
        with torch.no_grad():
            for idx, (imgs, label) in enumerate(selected_dataloader):
                #print('{}/{}'.format(idx, len(selected_dataloader)))


                #print(dataset.__getdetails__(idx))
                #sys.exit(0)
                im1, im2 = imgs
                ### Rewrite this!
                if tokens_from_disk:
                    token_distances = token_distances_dict[idx]
                else:
                    im1 = model.forward_features(im1)['x_norm_patchtokens'].squeeze(0)
                    im2 = model.forward_features(im2)['x_norm_patchtokens'].squeeze(0)

                #cd, wi = calc_distances(im1, im2, window_size=[8,8], method='paired')
                
                    token_distances = paired_distances(im1.cpu(), im2.cpu()).reshape(16, 16)
                    token_distances_dict[idx] = token_distances

                slide = np.lib.stride_tricks.sliding_window_view(token_distances, window_dim)
                window_size = window_dim[0] * window_dim[1]
                slide = slide.reshape(slide.shape[0], slide.shape[1], window_size)
                cd = np.mean(slide, axis=2)
                if adaptive_threshold:
                    selected_idxes = np.where(cd.flatten()>(threshold*(adaptive_threshold_dict[dataset.__getdetails__(idx)[0]])))[0]
                else:
                    selected_idxes = np.where(cd.flatten()>threshold)[0]
                
                pred_labels.append(min(1, len(selected_idxes)))
                gt_labels.append((int(label.cpu())))
        if not tokens_from_disk:    
            with open(token_distances_dict_filename, 'wb') as obj:
                pickle.dump(token_distances_dict, obj)
        
        pred_labels = np.array(pred_labels)
        gt_labels = np.array(gt_labels)

        acc = np.sum(np.where(pred_labels == gt_labels, 1, 0)/len(pred_labels))
        prec, rec, f1, fpr = calculate_metrics(gt_labels, pred_labels, extended=True)
        print('acc:', acc)
        print('prec:', prec)
        print('rec:', rec)
        print('f1:', f1)
        print('fpr:', fpr)
        

        if not args.roc:
            if not adaptive_threshold:
                score_df.loc[len(score_df)] = [window_dim, threshold, seed, acc, prec, rec, f1, latexify(acc, prec, rec, f1)]
            else:
                score_df.loc[len(score_df)] = [window_dim, 'adap_{}'.format(threshold), seed, acc, prec, rec, f1, latexify(acc, prec, rec, f1)]
            
        else:
            score_df.loc[len(score_df)] = [window_dim, threshold, seed, acc, prec, rec, f1, fpr, latexify(acc, prec, rec, f1)]
        score_df.to_csv(csv_name, index=False)




    ##imgs to tensor, to device
    ## fv = model.forward_features(x)['x_norm_patchtokens']
    ## fv = model.forward_features(x)['x_norm_patchtokens']
    ## fv = model.forward_features(x)['x_norm_patchtokens']

    ## Calc distances between 1-2 and 2-3
    ## Calc distance between "1-2" and "2-3"

    ## Make list of these distances

    # def calc_distances(self, patch_idxes, window_size=(1,1), method='paired'):
    # patch1_idxes = self.indices[patch_idxes[0]]
    # patch2_idxes = self.indices[patch_idxes[1]]
        
    #     if method=='paired':
    #         token_distances = paired_distances(self.CF.patches[patch1_idxes], self.CF.patches[patch2_idxes]).reshape(8, 25)
    #         window_dim = window_size[0] * window_size[1]
    #         slide = np.lib.stride_tricks.sliding_window_view(token_distances, window_size)
    #         patch1_window_indices = np.lib.stride_tricks.sliding_window_view(patch1_idxes.reshape(8, 25), window_size)
    #         patch2_window_indices = np.lib.stride_tricks.sliding_window_view(patch2_idxes.reshape(8, 25), window_size)
    #         window_indices = (patch1_window_indices, patch2_window_indices)
    #         slide = slide.reshape(slide.shape[0], slide.shape[1], window_dim)

    #         return np.mean(slide, axis=2), window_indices


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--input_dir', type=str, default='./')
    parser.add_argument('--pretrained', type=str, default='')
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--vanilla_posembed', type=int, default=0)
    parser.add_argument('--citypulse_root', type=str, default = '../../citypulse/')
    parser.add_argument('--window_dim', nargs='+', type=int, default=[8,8])
    parser.add_argument('--threshold', type=float, default=50)
    parser.add_argument('--object', type=str, default='balanced') #change this to boom if boom
    parser.add_argument('--relatexify', type=int, default=0)
    parser.add_argument('--adaptive_threshold', type=int, default=0)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--multiple_threshold', nargs='+', type=int, default=[0,0])
    parser.add_argument('--roc', type=int, default=0)
    args = parser.parse_args()
    
    main(args)

    