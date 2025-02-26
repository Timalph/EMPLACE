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
from utils import Tweaked_TripletMarginLoss_Linear, create_dir_name_ATM, Change_AMS, perform_forward_3, copy2xfs, copy2zfs, copy2zfs_mp
import sys
from transformers import ViTImageProcessor, ViTModel
from code_dinodino.build_dinodino import build_dinodino
## Whot can be passed as argument?

def main(args):
    seed=0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    pos_distance = args.pos
    neg_distance = args.neg
    batch_size = args.batch_size
    image_size = args.image_size
    patch_size = args.patch_size
    csv_file_TRAIN = args.csv_file_train
    csv_file_TEST = args.csv_file_test
    checkpoint = args.checkpoint
    total_epochs = args.total_epochs
    pos_prox = args.pos_prox
    neg_prox = args.neg_prox
    ### loading here to overwrite if loading a checkpoint
    best_acc = -1
    best_epoch = 0
    adjusted_batch_size = args.adjusted_batch_size
    encoder_architecture = args.encoder_architecture

    saved_models_dir = args.saved_models_dir
    if args.encoder_architecture == 'DINODINO':
        saved_models_dir = 'DINODINO_' + saved_models_dir + '/'
    if args.encoder_architecture == 'DINODINO14':
        saved_models_dir = 'DINODINO14_' + saved_models_dir + '/'
    if not args.ALLDATA:
        print('Running model with pos: {}\nneg: {}\nbatch_size: {}\nimage_size: {}\npatch_size: {}\ncsv_file: {}\nencoder_architecture: {}'.format(pos_distance, neg_distance, adjusted_batch_size if adjusted_batch_size else batch_size, image_size, patch_size, csv_file_TRAIN, encoder_architecture))
        print('Also running with ppx{} and npx{}'.format(pos_prox, neg_prox))
    else:
        print('RUNNING WITH ALL DATA BABYYYYY')
    dirpath = create_dir_name_ATM(args, saved_models_dir)
    #create_dir_name(args, saved_models_dir)
    os.makedirs(dirpath, exist_ok=True)
    print('Saving to:', dirpath)
    sleep(5)
    # #image_size = (640, 320)
    # print(dirpath)
    # if args.HDD:
    #     print('Copying to HDDStore...')
    #     os.system('cp -r {} /hddstore/talpher/{}'.format(args.input_dir, args.input_dir))
    #     hdd_dir = os.path.join('/hddstore/talpher/', args.input_dir)
    # else:
    #     print('Not running from HDDStore')
    #     hdd_dir = args.input_dir
    ## Check for csv

    ####################################################################### BUILD TRIPLET FILE #######################################################################
    if args.ALLDATA:
        triplet_file_TRAIN = args.csv_file_train
        triplet_file_TEST = args.csv_file_test
    else:
        if neg_prox != 9999:
            ## Train and test csvs
            if args.temp_congruent:
                raise NotImplementedError("did not fix temp congruent here yet")
            triplet_dir = 'triplets/triplet_ppx_{}_p{}_n{}_npx{}/'.format(pos_prox, pos_distance, neg_distance, neg_prox)
            os.makedirs(triplet_dir, exist_ok=True)
            triplet_file_TRAIN = 'triplets/triplet_ppx_{}_p{}_n{}_npx{}/train.csv'.format(pos_prox, pos_distance, neg_distance, neg_prox)
            triplet_file_TEST = 'triplets/triplet_ppx_{}_p{}_n{}_npx{}/test.csv'.format(pos_prox, pos_distance, neg_distance, neg_prox)
            if not os.path.isfile(triplet_file_TRAIN):
            #if not os.path.isfile(triplet_file):

                ### Read in train csv and test csv
                #TRAIN
                df = pd.read_csv(csv_file_TRAIN)
                ### Do all of this for both train and test
                print('filtering train distances for {} {} {} {}...'.format(pos_prox, pos_distance, neg_distance, neg_prox))
                filtered = gt.filter_distances_after_saving(df, pos_distance, neg_distance, pos_prox = pos_prox, neg_prox = neg_prox, step=1000000)
                filtered.to_csv(triplet_file_TRAIN)
                del df
                dataset_TRAIN=filtered
                #TEST
            else:
                dataset_TRAIN = pd.read_csv(triplet_file_TRAIN)

            if not os.path.isfile(triplet_file_TEST):
                df = pd.read_csv(csv_file_TEST)
                ### Do all of this for both train and test
                print('filtering test distances for {} {} {} {}...'.format(pos_prox, pos_distance, neg_distance, neg_prox))
                filtered = gt.filter_distances_after_saving(df, pos_distance, neg_distance, pos_prox = pos_prox, neg_prox = neg_prox, step=1000000)
                filtered.to_csv(triplet_file_TEST)
                del df
                dataset_TEST=filtered

            else:
                dataset_TEST = pd.read_csv(triplet_file_TRAIN)
        elif neg_prox == 9999:

            ## Train and test csvs
            triplet_ext = 'triplet_ppx_{}_p{}_n{}{}'.format(pos_prox, pos_distance, neg_distance, '_tcg' if args.temp_congruent else '')
            triplet_dir = 'triplets/{}/'.format(triplet_ext)
            triplet_file_TRAIN = 'triplets/{}/train.csv'.format(triplet_ext)
            triplet_file_TEST = 'triplets/{}/test.csv'.format(triplet_ext)
            os.makedirs(triplet_dir, exist_ok=True)
            if not os.path.isfile(triplet_file_TRAIN):
            #if not os.path.isfile(triplet_file):

                ### Read in train csv and test csv
                #TRAIN
                if args.temp_congruent:
                    df = pd.read_csv(csv_file_TRAIN[:-4] + '_tcg' + '.csv')
                else:
                    df = pd.read_csv(csv_file_TRAIN)
                ### Do all of this for both train and test
                print('filtering train distances for {} {} {}...'.format(pos_prox, pos_distance, neg_distance))
                filtered = gt.filter_distances_after_saving(df, pos_distance, neg_distance, pos_prox = pos_prox, step=1000000)
                filtered.to_csv(triplet_file_TRAIN)
                del df
                dataset_TRAIN=filtered
                #TEST
            else:
                dataset_TRAIN = pd.read_csv(triplet_file_TRAIN)

            if not os.path.isfile(triplet_file_TEST):
                if args.temp_congruent:
                    df = pd.read_csv(csv_file_TEST[:-4] + '_tcg' + '.csv')
                else:
                    df = pd.read_csv(csv_file_TEST)
                ### Do all of this for both train and test
                print('filtering test distances for {} {} {}...'.format(pos_prox, pos_distance, neg_distance))
                filtered = gt.filter_distances_after_saving(df, pos_distance, neg_distance, pos_prox = pos_prox, step=1000000)
                filtered.to_csv(triplet_file_TEST)
                del df
                dataset_TEST=filtered

            else:
                dataset_TEST = pd.read_csv(triplet_file_TRAIN)        
    ##################################################################################################################################################################
    try:
        print('Trying to open')
        mean, sd = pickle.load(open('./means/mean_sd_{}.pickle'.format(args.amount_of_clusters),'rb'))
    except FileNotFoundError:
        print('Finding mean and sd')
        start = time()
        mean, sd = calc_mean_and_sd(amount_of_clusters=args.amount_of_clusters)
        print(time() - start)
        with open('./means/mean_sd_{}.pickle'.format(args.amount_of_clusters), 'wb') as f:
            pickle.dump([mean, sd], f)

    ### If necessary, implement an argument for image size that can be passed to Change_AMS
    if args.XFS:
        copy2xfs(triplet_file_TRAIN, args.input_dir)
        copy2xfs(triplet_file_TEST, args.input_dir.replace('train', 'test'))
        hdd_dir  = os.path.join('/ivi/xfs/talpher/', args.input_dir)
    elif args.ZFS:
        copy2zfs_mp(triplet_file_TRAIN, args.input_dir, args.image_size)
        copy2zfs_mp(triplet_file_TEST, args.input_dir.replace('train', 'test'), args.image_size)
        hdd_dir  = os.path.join('/home/talpher/', args.input_dir)
    elif args.HDD_copy:
        print('Making a copy of the entire dataset in another resolution on the harddrive')
        copy2zfs_mp(triplet_file_TRAIN, args.input_dir, args.image_size, zfs_dir = '/home/talpher/phd/sscd/', skip_switch=True)
        copy2zfs_mp(triplet_file_TEST, args.input_dir.replace('train', 'test'), args.image_size, zfs_dir = '/home/talpher/phd/sscd/', skip_switch=True)
        print('Exiting...')
        print(sys.exit(0))
    else:
        hdd_dir = os.path.join('/home/talpher/phd/sscd/', args.input_dir)

    dataset = Change_AMS(triplet_file_TRAIN, hdd_dir, mean, sd, nojitter=args.nojitter)
    
    print('Dataset of size: {}'.format(dataset.__len__()))
    train_set_length = int(dataset.__len__() * 0.8)
    val_set_length = dataset.__len__() - train_set_length
    dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_set_length, val_set_length])
    print('Train set of size: {}\nTest set of size: {}'.format(train_set_length, val_set_length))

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    ### Load in dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device is: {}'.format(device))
    ### Load in model

    print('Loading model...')
    ### try with swin transformer

    #model = torchvision.models.vit_b_16(weights='IMAGENET1K_V1').to(device)
    #### image_size always (height, width)
    print(image_size)
    print(patch_size)

    if encoder_architecture == 'vit_b_16':
        model = VisionTransformer(image_size=image_size, patch_size=patch_size, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072)
        weight_url = 'https://download.pytorch.org/models/vit_b_16-c867db91.pth'
    elif encoder_architecture == 'vit_l_32':
        model = VisionTransformer(image_size=image_size, patch_size = patch_size, num_layers = 24, num_heads=16, hidden_dim=1024, mlp_dim=4096)
        weight_url = 'https://download.pytorch.org/models/vit_l_32-c7638314.pth' 
    elif encoder_architecture == 'DINOV2':
        model = ViTModel.from_pretrained('facebook/dino-vitb16')
    elif encoder_architecture =='DINODINO':
        model = build_dinodino()
    elif encoder_architecture =='DINODINO14':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    #elif encoder_architecture =='DINOV2_timm'

    print('Loaded model!')
    start_epoch=0
#    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    if args.loss == 'fixed_margin':
        criterium = Tweaked_TripletMarginLoss
    elif args.loss == 'adaptive':
        criterium = Tweaked_TripletMarginLoss_Adaptive
    elif args.loss == 'linear':
        criterium = Tweaked_TripletMarginLoss_Linear
    all_losses = []
    all_accs = []
    validation_acc = []
    all_train_accs = []
    print(dirpath)
    if checkpoint:
        print('Loading checkpoint {}'.format(checkpoint))

        checkpoint_dict = torch.load(os.path.join(dirpath, checkpoint))
        try:
            model.load_state_dict(checkpoint_dict['model_state_dict'])
        except RuntimeError:
            unparalleled_checkpoint_dict = {}
            for k, v in checkpoint_dict['model_state_dict'].items():
                print(k)
                unparalleled_checkpoint_dict[k[7:]] = v
            model.load_state_dict(unparalleled_checkpoint_dict)
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        start_epoch = checkpoint_dict['epoch']
        print('starting from epoch {}'.format(start_epoch))
        sleep(5)


        validation_acc_list = pickle.load(open(dirpath+'validation_acc.pickle', 'rb'))
        best_epoch = np.argmax(validation_acc_list)
        best_acc = np.max(validation_acc_list)
        loss = checkpoint_dict['loss']

        all_losses = pickle.load(open(os.path.join(dirpath, 'loss.pickle'), 'rb'))
        all_accs = pickle.load(open(os.path.join(dirpath, 'all_accs.pickle'), 'rb'))
        validation_acc = pickle.load(open(os.path.join(dirpath, 'validation_acc.pickle'), 'rb'))
        all_train_accs = pickle.load(open(os.path.join(dirpath, 'train_acc.pickle'), 'rb'))
    elif encoder_architecture == 'DINOV2' or encoder_architecture == 'DINODINO' or encoder_architecture=='DINODINO14':
        pass
    else:
        print('Using {} weights'.format(weight_url.split('/')[-1]))
        
        weights = torch.hub.load_state_dict_from_url(weight_url, progress = True)
        weight_dict = dict(weights)
        model_dict = dict(model.named_parameters())
        with open('vitweights.txt', 'w') as f:
            for line in list(weight_dict.keys()):
                f.write(line)
                f.write('\n')
        with open('modelweights.txt', 'w') as f:
            for line in list(model_dict.keys()):
                f.write(line)   
                f.write('\n')
        for model_key, vit_key in zip(model_dict.keys(), weight_dict.keys()):
            
            if model_dict[model_key].shape ==weight_dict[vit_key].shape:
                model_dict[model_key] = weight_dict[vit_key]
            else:
                print('{} could not be loaded due to differing shapes:'.format(model_key))
                print(model_dict[model_key].shape, weight_dict[vit_key].shape)
        # optimizer = adam
        print('before: {}'.format(sum_params(model)))
        model.load_state_dict(model_dict)
        print('after: {}'.format(sum_params(model)))

    if not torch.cuda.is_available():
        print('No GPU')
        sys.exit(0) 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    if args.multigpu:   
        model = nn.DataParallel(model, device_ids=list(range(args.multigpu)))
        model.to(device)
    else:
        model = model.to(device)
    #model.to(f'cuda:{model.device_ids[0]}')
    #model.to(gpu_ids[0])
    #device = gpu_ids[0]
    model.train()

    early_stopping_threshold = args.early_stopping_threshold

    for epoch in range(start_epoch, start_epoch + total_epochs):
        train_accs = []
        epoch_loss = 0
        start_epoch_time = time()
        startstep=time()
        for idx, (triplet_imgs, triplet_labels) in enumerate(train_dataloader):
            #sys.exit(0)
            #print('{}/{}'.format(idx, train_dataloader.__len__()))
            anc, pos, neg = triplet_imgs
            #t = torch.stack((anc, pos, neg), dim=1).squeeze(0)
            
            anc = anc.to(device)
            pos = pos.to(device)
            neg = neg.to(device)
            triplet_labels = triplet_labels.to(device)

            anchor_out, positive_out, negative_out = perform_forward_3(model, anc, pos, neg, args.encoder_architecture)

            loss, acc = criterium(anchor_out, positive_out, negative_out, triplet_labels)
            epoch_loss   += float(loss.detach().cpu().numpy())
            loss.backward()
            
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
                cutint = np.random.randint(0, anc.shape[2])
                anc = cut_and_flip(anc, cutint)
                pos = cut_and_flip(pos, cutint)
                neg = cut_and_flip(neg, cutint)
                anchor_out, positive_out, negative_out = perform_forward_3(model, anc, pos, neg, args.encoder_architecture)
                loss, acc = criterium(anchor_out, positive_out, negative_out, triplet_labels)
                epoch_loss += float(loss.detach().cpu().numpy())
                loss.backward()
                #nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                #optimizer.step()

                loss = float(loss.detach().cpu().numpy())
                acc  = float(acc.detach().cpu().numpy())
                print('Loss:', loss)
                print('Acc:', acc)
                print('Time:', time()-startstep)
                startstep=time()
                print('{}/{}'.format(idx, train_dataloader.__len__()))  
                all_losses.append(loss)
                train_accs.append(acc)
            if adjusted_batch_size:
                if (idx+1) % int(adjusted_batch_size/batch_size) == 0 or (idx+1) == train_dataloader.__len__():
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    if idx < 100:
                        print('Did a step after {} images'.format(batch_size*(idx+1)))
            else:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            if args.ALLDATA:
                if idx%1000 == 0:
                    PATH = dirpath + 'tmp_epoch.pth'
                    torch.save({
                        'epoch': epoch,
                        'idx' : idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        }, PATH)
        print('{}/{}, loss:{}'.format(epoch, 30, epoch_loss))
        print(time() - start_epoch_time)

        ### Test
        batch_acc = []
        model.eval()
        with torch.no_grad():
            for idx, (triplet_imgs, triplet_labels) in enumerate(val_dataloader):
                del anc, pos, neg, anchor_out, positive_out, negative_out
                #print('{}/{}'.format(idx, train_dataloader.__len__()))
                anc, pos, neg = triplet_imgs
                #t = torch.stack((anc, pos, neg), dim=1).squeeze(0)
                
                anc = anc.to(device)
                pos = pos.to(device)
                neg = neg.to(device)
                triplet_labels = triplet_labels.to(device)

                anchor_out, positive_out, negative_out = perform_forward_3(model, anc, pos, neg, args.encoder_architecture)

                _, acc = criterium(anchor_out, positive_out, negative_out, triplet_labels)

                acc  = float(acc.cpu().numpy())
                print('Acc:', acc)
                print('Time:', time()-startstep)
                startstep=time()
                print('Testing {}/{}'.format(idx, val_dataloader.__len__()))  
                batch_acc.append(acc)
        epoch_accuracy = np.mean(batch_acc)
        validation_acc.append(epoch_accuracy)
        all_accs += train_accs
        all_train_accs.append(np.mean(train_accs))
        with open(dirpath+'validation_acc.pickle', 'wb') as pfile:
            pickle.dump(validation_acc, pfile)
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
            best_epoch = epoch
            PATH = dirpath + 'best_epoch.pth'.format(epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, PATH)
            best_acc = epoch_accuracy
        elif epoch - best_epoch > early_stopping_threshold-1:
            print("Training stopped at epoch {} early due to not improving on the validation accuracy for {} epochs".format(epoch, early_stopping_threshold))
            break

#anchor
#positive
#negative

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', type=str, default='./')
    parser.add_argument('--csv_file_train', type=str, default='AC-1M_train.csv')
    parser.add_argument('--csv_file_test', type=str, default='AC-1M_test.csv')
    parser.add_argument('--batch_size', type=int, default = '1')
    parser.add_argument('--multigpu', type=int, default = '0')
    parser.add_argument('--adjusted_batch_size', type=int, default = '0')
    parser.add_argument('--pos', type=int, default=365)
    parser.add_argument('--neg', type=int, default=800)
    parser.add_argument('--patch_size', type=int, default=100)
    parser.add_argument('--amount_of_clusters', type=int, default=10000)
    parser.add_argument('--image_size', nargs='+', type=int, default=[1280,4000])
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--total_epochs', type=int, default=20)
    parser.add_argument('--pos_prox', type=int, default=0)
    parser.add_argument('--neg_prox', type=int, default=9999)
    parser.add_argument('--early_stopping_threshold', type=int, default=5)
    parser.add_argument('--cutandflip', type=int, default=0)
    parser.add_argument('--XFS', type=int, default=0)
    parser.add_argument('--ZFS', type=int, default=0)
    parser.add_argument('--HDD', type=int, default=1)
    parser.add_argument('--gpuid', nargs=1, type=str, default='0') #python main.py --gpuid=0,1,2,3
    parser.add_argument('--temp_congruent', type=int, default=0) ## Do you want anc pos neg to be in the same direction through time
    parser.add_argument('--nojitter', type=int, default=1)
    parser.add_argument('--encoder_architecture', type=str, default = "vit_b_16")
    parser.add_argument('--headtype', type=str, default='attention')
    parser.add_argument('--grad_clip', type=float, default=5)
    parser.add_argument('--loss', type=str, default='fixed_margin')
    parser.add_argument('--saved_models_dir', type=str, default='ATM_saved_models')
    parser.add_argument('--ALLDATA', type=int, default=0)
    parser.add_argument('--HDD_copy', type=int, default=0)
    
    args = parser.parse_args()
    
    main(args)
