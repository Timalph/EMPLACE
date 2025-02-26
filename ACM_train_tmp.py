#import
#import rectangular vit
from utils import CityPulsePano
from utils import create_dir_name
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
import gc
import generate_triplets as gt
from Rectangular_VIT import VisionTransformer
import random
from torchvision.utils import save_image
from adjusted_factory import create_vit
import sys
from utils import cut_and_flip, copy2zfs_mp
from utils import location_sampling
from utils import perform_forward_2
from transformers import ViTImageProcessor, ViTModel
from transformers import CLIPProcessor, CLIPModel
#from code_dinodino.build_dinodino import build_dinodino
### Use same optimizer and clipping as Citypulse


def main(args):

    #if args.cls_style:
    #    raise NotImplementedError("MAKE SURE YOU SAVE TO CLS DIRECTORY")
    seed=args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if not torch.cuda.is_available():
        sys.exit(0)

    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    ### DATASET ###########################################################################################
    mean, sd = pickle.load(open('./means/mean_sd_{}.pickle'.format(args.amount_of_clusters),'rb'))


    if args.ATM_buildings:
        ### COPY EVERYTHING TO TEST BECAUSE THIS IS ALL SOURCED FROM THE TEST SET
        destination = 'antenna_{}_{}_test'.format(args.image_size[1], args.image_size[0])   
        train_df = 'ATM_buildings_trainset.csv'
        copy2zfs_mp(train_df, destination, args.image_size)
        test_df = 'ATM_buildings_testset.csv'
        copy2zfs_mp(test_df, destination, args.image_size)
        val_df = 'ATM_buildings_valset.csv'
        copy2zfs_mp(val_df, destination, args.image_size)
    elif args.ATM_buildings_square:
        normalized_splits = np.array(args.splits)/100
        splits = '{}train_{}_val_{}_test_{}'.format('{}_'.format(args.balanced) if args.balanced else '',normalized_splits[0], normalized_splits[1], normalized_splits[2])
        train_df =  'ATM_square_trainsplits/ATM_buildings_square_{}_{}_trainset.csv'.format(args.ATM_buildings_square, splits)      
        test_df  =  'ATM_square_trainsplits/ATM_buildings_square_{}_{}_testset.csv'.format(args.ATM_buildings_square, splits)
        val_df   =  'ATM_square_trainsplits/ATM_buildings_square_{}_{}_valset.csv'.format(args.ATM_buildings_square, splits)

    else:
        train_df = 'ACM_trainset_clusters_{}.csv'.format(args.labels)
        val_df = 'ACM_valset_clusters_{}.csv'.format(args.labels)
    print('?')
    #args.citypulse_root = '../../citypulse/'
    trainset = CityPulsePano(train_df, args.citypulse_root, mean, sd, device=device, nojitter=args.nojitter, softmax=args.softmax)
    valset = CityPulsePano(val_df, args.citypulse_root, mean, sd, device=device, nojitter=args.nojitter, softmax=args.softmax)

    #print('Dataset of size: {}'.format(dataset.__len__()))
    #train_set_length = int(dataset.__len__() * args.trainsplit)
    #val_set_length = dataset.__len__() - train_set_length
    #trainset, valset = torch.utils.data.random_split(dataset, [train_set_length, val_set_length])
    print('Train set of size: {}\nval set of size: {}'.format(len(trainset), len(valset)))

    train_dataloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, worker_init_fn=seed)
    val_dataloader = DataLoader(valset,batch_size=1)

    #######################################################################################################

    ### MODEL
    if args.headtype=='DINOV2':
        model = ViTModel.from_pretrained('facebook/dino-vitb16')
    #elif args.headtype=='DINODINO':
    #    model = build_dinodino()
    elif args.headtype=='resnet':
        model = torch.hub.load("pytorch/vision:v0.13.0", "resnet50", weights="IMAGENET1K_V2")
    elif args.headtype=='clip':
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    elif args.headtype=='DINODINO14' or args.headtype=='DINODINO':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    else:
        model = VisionTransformer(image_size=args.image_size, patch_size=args.patch_size, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072)
        weight_url = 'https://download.pytorch.org/models/vit_b_16-c867db91.pth'
    ############################################################ LOAD PRETRAINED ##################################################################
    if args.pretrained and args.headtype=='DINODINO':
        checkpoint_dict = torch.load(args.pretrained)
        newdict = {}
        for k, v in checkpoint_dict['model'].items():
            newkey = k.replace('student.backbone.', '')
            newkey = newkey.replace('blocks.0', 'blocks')
            newkey = newkey.replace('teacher.backbone.', '')
            #if k[:25] == 'student.backbone.blocks.0':
            #    newdict[k[25:]] = v
            newdict[newkey] = v
        model.load_state_dict(newdict, strict=False)
    elif args.pretrained and args.headtype == 'DINODINO14':
        checkpoint_dict = torch.load(args.pretrained)
        if args.vanilla_posembed:
            try:
                checkpoint_dict['model_state_dict'].pop('pos_embed')
            except:
                checkpoint_dict['model_state_dict'].pop('module.pos_embed')

            print('RUNNING WITH VANILLA POSITIONAL EMBEDDING')
        if 'module.norm.bias' in checkpoint_dict['model_state_dict'].keys():
            unparalleled_checkpoint_dict = {}
            for k, v in checkpoint_dict['model_state_dict'].items():
                print(k)
                unparalleled_checkpoint_dict[k[7:]] = v
            model.load_state_dict(unparalleled_checkpoint_dict, strict=not args.vanilla_posembed)
            print('loaded pretrained from multi')
        else:
            model.load_state_dict(checkpoint_dict['model_state_dict'], strict=not args.vanilla_posembed)
            print('loaded pretrained from single')
        print('DINODINO14 loaded!')
        if args.freeze:
            for param in model.parameters():
                param.requires_grad = False
    elif args.pretrained and not args.headtype=='DINODINO':
        print('Loading pretrained {}'.format(args.pretrained))
        checkpoint_dict = torch.load(args.pretrained)
        unparalleled_checkpoint_dict = {}
        try:
            for k, v in checkpoint_dict['model_state_dict'].items():
                unparalleled_checkpoint_dict[k[7:]] = v
            model.load_state_dict(unparalleled_checkpoint_dict)
        except RuntimeError:
            model.load_state_dict(checkpoint_dict['model_state_dict'])  
        if args.freeze:
            for param in model.parameters():
                param.requires_grad = False
    elif args.headtype!= 'DINOV2' and args.headtype != 'DINODINO14' and args.headtype != 'resnet' and args.headtype != 'clip':
        print('Using {} weights'.format(weight_url.split('/')[-1]))
        
        weights = torch.hub.load_state_dict_from_url(weight_url, progress = True)
        weight_dict = dict(weights)
        model_dict = dict(model.named_parameters())
        ### I believe weights are being replaced here instead of filled in. Maybe only replace weights if the shape is the same??
        for model_key, vit_key in zip(model_dict.keys(), weight_dict.keys()):
            if model_dict[model_key].shape ==weight_dict[vit_key].shape:
                model_dict[model_key] = weight_dict[vit_key]
            else:
                print('{} could not be loaded due to differing shapes:'.format(model_key))
                print(model_dict[model_key].shape, weight_dict[vit_key].shape)
    ##############################################################################################################################################
    # eigenlijk 768
    #### We have input_dim, headtype   

    #write create_model in utils
    #model = create_model(input_dim, model, headtype)
    if args.headtype == 'cls' or args.headtype=='DINOV2' or args.headtype=='DINODINO' or args.headtype=='DINODINO14':
        input_dim = 768
    elif args.headtype == 'attention' or args.headtype == 'resnet':
        input_dim = 1000
    elif args.headtype == 'clip':
        input_dim = 512
    else:
        raise NotImplementedError('Only \'cls\', \'DINOV2\', \'DINODINO(14)\' or \'attention\' are valid options')
    if args.freeze:
        for param in model.parameters():
            param.requires_grad = False

    model.to(device)
    model.train()

    if not args.softmax:
        if args.cls_style:
            head = torch.nn.Linear(1 + (input_dim*3), 1, bias=True if args.headbias else False)
        elif args.cls_only:
            head = torch.nn.Linear(1, 1, bias=True if args.headbias else False)
        elif args.triplet_train==1:
            head = torch.nn.Linear(2, 1, bias=True if args.headbias else False)
        else:
            head = torch.nn.Linear(input_dim*3, 1, bias=True if args.headbias else False)
        sigmoid_or_softmax = torch.nn.Sigmoid()
        #riterium = torch.nn.BCELoss()
        criterium = torch.nn.BCEWithLogitsLoss()


    elif args.softmax:
        head = torch.nn.Linear(input_dim*3, 2, bias=True if args.headbias else False)
        sigmoid_or_softmax = torch.nn.Softmax(dim=1)
        criterium = torch.nn.CrossEntropyLoss()
    if args.triple_head:
            head = torch.nn.Sequential(torch.nn.Linear(input_dim*3, input_dim, bias=True if args.headbias else False),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(input_dim, 256, bias=True if args.headbias else False),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(256, 2, bias=True if args.headbias else False))
    head = head.to(device)

    try:
        head.load_state_dict(checkpoint_dict['head'])
        print('loaded head')
    except KeyError:
        print('cant load head')
    except UnboundLocalError:
        print('running vanilla!')
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Bookkeeping#################################################################################################################################################

    start_epoch = 0
    stop_epochs = args.stop_epochs

    all_losses = []
    all_accs = []
    validation_acc = []
    all_train_accs = []

    best_acc = -1
    best_epoch = 0
    early_stopping_threshold = args.early_stopping_threshold

    if args.ATM_buildings:
        args.saved_models_dir = 'ATM_buildings_models/'
        if args.cls_distribution:
            args.saved_models_dir = 'CLS_distances/ATM_buildings/'
    if args.ATM_buildings_square:
        args.saved_models_dir = 'ATM_buildings_square_models_{}/{}/'.format(args.ATM_buildings_square, splits)

    saved_models_dir = args.saved_models_dir

    #saved_models_dir = './citypulse_models/'
    dirpath = create_dir_name(args, saved_models_dir)
    print('Saving to:', dirpath)
    os.makedirs(dirpath, exist_ok=True)
    pdist = nn.PairwiseDistance(p=2)
    ################################################################### BUILD CLS DISTRIBUTION ####################################################################
    if args.cls_distribution:
        print('Building CLS Distribution')

        blueprintfile = 'ACM_{}set_clusters_{}.csv' if not args.ATM_buildings else 'ATM_buildings_{}set.csv'
        model.eval()
        all_cls_distances = []
        with torch.no_grad():
            for split in ['train', 'val', 'test']:
                split_cls_distances = []
                dataset_file = blueprintfile.format(split, args.labels)
                dataset = CityPulsePano(dataset_file, args.citypulse_root, mean, sd, device=device, nojitter=args.nojitter, softmax=args.softmax)
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
                for idx, (imgs, label) in enumerate(dataloader):
                    print('{} at {}%, (size {})'.format(split, np.round(idx/dataloader.__len__()*100,1), dataloader.__len__()))
                    im1, im2  = imgs
                    out1, out2= perform_forward_2(model, im1, im2, args.headtype)
                    ## Add the eucl. dist. of the cls tokens together with the changeflag
                    cls_dist = pdist(out1, out2).cpu().numpy()[0]
                    label = label.cpu().numpy()[0]
                    split_cls_distances.append([cls_dist, label])
                    all_cls_distances.append([cls_dist, label])
                with open(dirpath+'{}_cls_distances.pickle'.format(split), 'wb') as f:
                    pickle.dump(np.array(split_cls_distances), f)
        with open(dirpath+'all_cls_distances.pickle', 'wb') as f:
            pickle.dump(np.array(all_cls_distances), f)
        print('Retrieved all cls distances!')
        sys.exit(0)
    ##############################################################################################################################################################

    for epoch in range(start_epoch, start_epoch + stop_epochs):
        train_accs = []
        epoch_loss = 0
        start_epoch_time = time()
        startstep=time()
        for idx, (imgs, label) in enumerate(train_dataloader):
            im1, im2  = imgs

            out1, out2= perform_forward_2(model, im1, im2, args.headtype)
            # if args.headtype == 'DINOV2':
            #     out1 = model.forward(im1, interpolate_pos_encoding=True)[1]
            #     out2 = model.forward(im2, interpolate_pos_encoding=True)[1]
            # elif args.headtype == 'DINODINO':
            #     out1 = model.forward(im1)
            #     out2 = model.forward(im2)
            # else:
            #     out1 = model.forward(im1, args.headtype)
            #     out2 = model.forward(im2, args.headtype)
            #print(pdist(out1, out2).unsqueeze(1).shape)
            #print(out1.shape)
            if args.cls_style:
                x = torch.concat([out1, out2, out2-out1, pdist(out1, out2).unsqueeze(1)], dim=1)      
            if args.cls_only:
                x = pdist(out1, out2).unsqueeze(1)
            else:
                x = torch.concat([out1, out2, out2-out1], dim=1)  

            pred = head(x)
            if args.softmax:
                loss = criterium(pred, label)
            else:
                loss = criterium(pred.view(-1), label)
            epoch_loss += float(loss.detach().cpu().numpy())
            loss.backward()

            #nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip/10)
            #optimizer.step()

            if args.softmax:
                acc_preds = torch.argmax(pred, dim=1).float()
            else:   
                acc_preds = (pred.view(-1)>0.5).float()
            acc = torch.sum(torch.where(acc_preds==label, 1, 0))/len(acc_preds)

            loss = float(loss.detach().cpu().numpy())
            acc  = float(acc.detach().cpu().numpy())
            print('Loss:', loss)
            print('Acc:', acc)
            print('Time:', time()-startstep)
            startstep=time()
            print('{}/{}'.format(idx, train_dataloader.__len__()))  
            all_losses.append(loss)
            train_accs.append(acc)

            if args.cutandflip:
                cutint = np.random.randint(0, im1.shape[2])
                im1 = cut_and_flip(im1, cutint)
                im2 = cut_and_flip(im2, cutint)

                out1, out2 = perform_forward_2(model, im1, im2, args.headtype)

                if args.cls_style:
                    x = torch.concat([out1, out2, out2-out1, pdist(out1, out2).unsqueeze(1)], dim=1)      
                if args.cls_only:
                    x = pdist(out1, out2).unsqueeze(1)
                else:
                    x = torch.concat([out1, out2, out2-out1], dim=1)    

                pred = head(x)

                if args.softmax:
                    loss = criterium(pred, label)
                else:
                    loss = criterium(pred.view(-1), label)



                epoch_loss += float(loss.detach().cpu().numpy())
                loss.backward()

                #nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip/10)
                #optimizer.step()
                if args.softmax:
                    acc_preds = torch.argmax(pred, dim=1).float()
                else:   
                    acc_preds = (pred.view(-1)>0.5).float()
                acc = torch.sum(torch.where(acc_preds==label, 1, 0))/len(acc_preds)

                loss = float(loss.detach().cpu().numpy())
                acc  = float(acc.detach().cpu().numpy())
                print('Loss:', loss)
                print('Acc:', acc)
                print('Time:', time()-startstep)
                startstep=time()
                print('{}/{}'.format(idx, train_dataloader.__len__()))  
                all_losses.append(loss)
                train_accs.append(acc)
            
            if args.adjusted_batch_size:
                if (idx+1) % int(args.adjusted_batch_size/args.batchsize) == 0 or (idx+1) == train_dataloader.__len__():
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip/10)
                    optimizer.step()
                    if idx < 100:
                        print('Did a step after {} images'.format(args.batchsize*(idx+1)))
            else:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip/10)
                optimizer.step()

        print('{}/{}, loss:{}'.format(epoch, 30, epoch_loss))
        print(time() - start_epoch_time)            
        #### Eval
        batch_acc = []
        model.eval()
        with torch.no_grad():
            all_preds=[]
            for idx, (imgs, label) in enumerate(val_dataloader):
                #del im1, im2, out1, out2
                im1, im2 = imgs

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
                print(acc_preds)
                acc = torch.sum(torch.where(acc_preds==label, 1, 0))/len(acc_preds)
                acc  = float(acc.detach().cpu().numpy())
                print('Acc:', acc)
                print('Time:', time()-startstep)
                startstep=time()
                print('Testing {}/{}'.format(idx, val_dataloader.__len__()))  
                batch_acc.append(acc)
                all_preds.append(int(acc_preds.detach().cpu().numpy()))
        epoch_accuracy = np.mean(batch_acc)
        validation_acc.append(epoch_accuracy)
        all_accs += train_accs
        all_train_accs.append(np.mean(train_accs))
        with open(dirpath+'validation_acc.pickle', 'wb') as pfile:
            pickle.dump(validation_acc, pfile)
        with open(dirpath+'validation_preds_{}.pickle'.format(epoch), 'wb') as pfile:
            pickle.dump(all_preds, pfile)
        model.train()
        with open(dirpath + 'time.txt', 'a') as f:
            f.write('Epoch {} took {} and got {:0.8f} train_acc and {:0.8f} validation_acc'.format(epoch, time() - start_epoch_time, np.mean(train_accs), epoch_accuracy))
            f.write('\n')
        with open(dirpath+'loss.pickle', 'wb') as pfile:
            pickle.dump(all_losses, pfile)
        with open(dirpath+'all_accs.pickle', 'wb') as pfile:
            pickle.dump(all_accs, pfile)
        with open(dirpath+'train_acc.pickle', 'wb') as pfile:
            pickle.dump(all_train_accs, pfile)

        print('epoch:', epoch, 'best epoch:', best_epoch)
        print(epoch_accuracy, best_acc)
        if epoch_accuracy > best_acc + 0.001:
            print('Saving epoch {} with accuracy {} as best epoch!'.format(epoch, epoch_accuracy))
            with open(dirpath+'all_val_preds.pickle', 'wb') as pfile:
                pickle.dump(all_preds, pfile)            
            best_epoch = epoch
            PATH = dirpath + 'best_epoch.pth'.format(epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'head': head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, PATH)
            best_acc = epoch_accuracy
        elif epoch - best_epoch > early_stopping_threshold-1:
            print("Training stopped at epoch {} early due to not improving on the validation accuracy for {} epochs".format(epoch, early_stopping_threshold))
            break
    print(dirpath)
    
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    command = 'python ACM_test.py --headtype={} --dirpath={} --citypulse_root={} --image_size {} {} --patch_size={} --testfile=ACM_testset_clusters_360.csv{}{}{}{}{}{}{}{}'.format(args.headtype, 
    dirpath, args.citypulse_root, args.image_size[0], args.image_size[1], args.patch_size, ' --ATM_buildings=1' if args.ATM_buildings else '', ' --ATM_buildings_square={}'.format(args.ATM_buildings_square) if args.ATM_buildings_square else '',
    ' --splits {} {} {}'.format(args.splits[0], args.splits[1], args.splits[2]) if args.ATM_buildings_square else '', ' --resultfile={}'.format(args.resultfile) if args.resultfile else '', ' --cls_style=1' if args.cls_style else '',
    ' --cls_only=1' if args.cls_only else '', ' --balanced={}'.format(args.balanced), ' --seed={}'.format(args.seed))
    print(command)
    os.system(command)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./')
#    parser.add_argument('--trainfile', type=str, default='ACM_trainset.csv')
    parser.add_argument('--labels', type=str, default='original')
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--batchsize', type=int, default =8)
    parser.add_argument('--adjusted_batch_size', type=int, default=0)
    parser.add_argument('--grad_clip', type=int, default=5) # .5
    parser.add_argument('--stop_epochs', type=int, default=20)
    parser.add_argument('--image_size', nargs='+', type=int, default=[128,400])
    parser.add_argument('--amount_of_clusters', type=int, default=10000)
    parser.add_argument('--early_stopping_threshold', type=int, default=5)
    parser.add_argument('--cutandflip', type=int, default=0)
    parser.add_argument('--nojitter', type=int, default=0)
    parser.add_argument('--pretrained', type=str, default='')
    parser.add_argument('--citypulse_root', type=str, default = '../../citypulse/')
    parser.add_argument('--freeze', type=int, default=1)
    parser.add_argument('--headtype', type=str, default='attention')
    parser.add_argument('--headbias', type=int, default=1)
    parser.add_argument('--saved_models_dir', type=str, default='./citypulse_models/')
    parser.add_argument('--resultfile', type=str, default='')
    parser.add_argument('--trainsplit', type=float, default=.9)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--softmax', type=int, default=0)
    parser.add_argument('--triple_head', type=int, default=0)
    parser.add_argument('--one_epoch', type=int, default=0)
    parser.add_argument('--ATM_buildings', type=int, default=0)
    parser.add_argument('--ATM_buildings_square', type=str, default='') #masked, regular, equirectangular
    parser.add_argument('--cls_style', type=int, default=0)
    parser.add_argument('--cls_only', type=int, default=0)
    parser.add_argument('--cls_distribution', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--splits', nargs='+', type=int)
    parser.add_argument('--balanced', type=str, default='') 
    parser.add_argument('--vanilla_posembed', type=int, default=0)
    parser.add_argument('--triplet_train', type=int, default=0)
    
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




