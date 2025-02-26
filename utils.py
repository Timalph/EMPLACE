import numpy as np
import os
from torchvision import transforms
from PIL import Image
from time import sleep
from time import time
import argparse
import pickle
import pandas as pd
import torch
from torch import nn
from datetime import timedelta
from datetime import date
import sys
from PIL import UnidentifiedImageError
from tqdm import tqdm
import multiprocessing


def imgtotensor(i, mean, sd, nojitter=False):
    if nojitter:
        transform =   transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, sd)])
        t = transform(i)
    else:
        transform =   transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, sd)])
        t = transform(i)
    return t

class CityPulsePano():
    def __init__(self, csv_file, path, mean, sd, device='cpu', nojitter=False, softmax=False):
        try:
            self.dataset = pd.read_csv(csv_file)
        except TypeError:
            #if passed a df
            self.dataset = csv_file
        self.path = path
        self.mean = mean
        self.sd = sd
        self.nojitter = nojitter
        self.device = device
        self.softmax = softmax
    def __len__(self):
        return len(self.dataset)
    def __getdetails__(self, idx):
        row = self.dataset.iloc[idx]
        return row['cluster'], row['change_flag']
    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        names = ['im1', 'im2']
        label = row['change_flag']

        try:
            imgs = [Image.open(os.path.join(self.path, row[name])) for name in names]
        except UnidentifiedImageError:
            imgs = [Image.open(os.path.join(self.path.replace('/home/talpher/', ''), row[name])) for name in names]
        if not self.softmax:
            label = torch.tensor(label).to(self.device).int().float()
        else:
            label = torch.tensor(label).to(self.device).long()
        return [imgtotensor(x, self.mean, self.sd, nojitter=self.nojitter).to(self.device) for x in imgs], label


def create_dir_name(args, saved_models_dir):
    if args.headtype == 'DINODINO':
        dinomodel = args.pretrained.split('/')[-1]
        dinodetails = dinomodel.split('.')
        return saved_models_dir + '/{}_{}/'.format(dinodetails[0], dinodetails[1])

    config_dir = '/{}_gc_{}{}{}{}{}{}{}{}{}{}{}{}{}/'.format(args.headtype,
                                        args.grad_clip,
                                        '_cnf' if args.cutandflip else '',
                                        '_nojitter' if args.nojitter else '', 
                                        '_nofreeze' if not args.freeze else '',
                                        '_tripleh' if args.triple_head else '',
                                        '_nhbias' if not args.headbias else '',
                                        '' if args.trainsplit==.9 else '_{}'.format(args.trainsplit),
                                        '' if args.lr ==.00001 else '_{}'.format(args.lr),
                                        '' if not args.softmax else '_smax',
                                        '' if not args.adjusted_batch_size else 'ab_{}'.format(args.adjusted_batch_size),
                                        '' if not args.seed else '_seed{}'.format(args.seed),
                                        '' if not args.cls_only else '_cls_only',
                                        '' if not args.triplet_train else 'triplet_train{}'.format(args.triplet_train))
    dirpath = saved_models_dir + 'img_{}_{}_batch_{}_patch_{}'.format(args.image_size[0], args.image_size[1], 
            args.batchsize, args.patch_size)
    if args.pretrained:
        pretrained_dir = args.pretrained.split('/')[-2]
        if not 'citypulse_models' in args.pretrained:
            assert 'ppx' in pretrained_dir
        else:
            pretrained_dir = 'citypulse_models_' + pretrained_dir   
        topdir = args.pretrained.split('/')[-3]
#        dirpath = saved_models_dir + pretrained_dir[:28]
        dirpath = saved_models_dir + topdir + '/' + pretrained_dir + '{}{}{}{}{}{}{}{}{}{}/'.format('_cnf' if args.cutandflip else '',
                                                                                    '_nofreeze' if not args.freeze else '',
                                                                                    '_tripleh' if args.triple_head else '',
                                                                                    '_nhbias' if not args.headbias else '',
                                                                                    '' if args.trainsplit==.9 else '_{}'.format(args.trainsplit),
                                                                                    '' if args.lr ==.00001 else '_{}'.format(args.lr),
                                                                                    '' if not args.softmax else '_smax',
                                                                                    '' if not args.adjusted_batch_size else 'ab_{}'.format(args.adjusted_batch_size),
                                                                                    '' if not args.seed else '_seed{}'.format(args.seed),
                                                                                    '' if not args.cls_only else '_cls_only',
                                                                                    '' if not args.triplet_train else 'triplet_train{}'.format(args.triplet_train))#[:28]
        #config_dir = '/' + pretrained_dir[29:] + '_' + config_dir[1:]
        #config_dir = '_' + config_dir[1:]
        print(dirpath)
        return dirpath

        #print('saving to:', dirpath+config_dir)
    return dirpath + config_dir

def create_dir_name_ATM(args, saved_models_dir):
    print('tcg:', args.temp_congruent)
    if args.adjusted_batch_size:
        batch_size = args.adjusted_batch_size 
    else:
        batch_size = args.batch_size
    dirpath = saved_models_dir + 'img_{}_{}_batch_{}_patch_{}{}{}{}'.format(args.image_size[0], 
                                                                    args.image_size[1],
                                                                    batch_size, 
                                                                    args.patch_size,
                                                                    '_nojitter' if args.nojitter else '_jitter',
                                                                    '_tcg' if args.temp_congruent else '_ncg',
                                                                    '_nocnf' if not args.cutandflip else '')

    config_dir = '/{}_ppx{}_p{}_n{}{}{}{}{}/'.format(args.headtype,
                                                args.pos_prox, 
                                                args.pos, 
                                                args.neg, 
                                                '_npx'.format(args.neg_prox) if not args.neg_prox == 9999 else '',
                                                '_gc{}'.format(args.grad_clip),
                                                '_{}'.format(args.encoder_architecture),
                                                '_{}_loss'.format(args.loss))

    if args.ALLDATA:
        config_dir = '/{}_{}{}{}{}/'.format(args.headtype,
                                                'ALLDATA',
                                                '_gc{}'.format(args.grad_clip),
                                                '_{}'.format(args.encoder_architecture),
                                                '_{}_loss'.format(args.loss))

    return dirpath + config_dir 

def cut_and_flip(x, randint):
    
    left = x[:,:,:randint]
    right = x[:,:,randint:]
    
    return torch.cat((right,left), dim=2)

def calc_mean_and_sd(d='antenna_4k_12/*', amount_of_clusters=30):
    print('Calulating mean and sd with {} random clusters...'.format(amount_of_clusters))
    transform = transforms.Compose([transforms.ToTensor()])
    image_paths = []
    stacked_tensors = False
    for idx, nb in enumerate(glob(d)):
        print(nb, '{}/{}'.format(idx, len(glob(d))))
        clusters = glob(nb + '/*')
        random.shuffle(clusters)
        selected_clusters = clusters[:amount_of_clusters]
        for cluster in selected_clusters:
            image_paths += glob(cluster + '/*')
    r2 = 0
    g2 = 0
    b2 = 0
    r_sd = 0
    g_sd = 0
    b_sd = 0
    for idx, path in enumerate(image_paths):
        if idx%10 == 0:
            print('{}/{}'.format(idx, len(image_paths)))
        img = Image.open(path)
        t = transform(img)
        r2 += t[0,:,:].mean()
        g2 += t[1,:,:].mean()
        b2 += t[2,:,:].mean()
        r_sd += t[0,:,:].std()
        g_sd += t[1,:,:].std()
        b_sd += t[2,:,:].std()
    r2 = r2/len(image_paths)
    g2 = g2/len(image_paths)
    b2 = b2/len(image_paths)
    r_sd = r_sd/len(image_paths)
    g_sd = g_sd/len(image_paths)
    b_sd = b_sd/len(image_paths)
    mean = (r2, g2, b2)
    std  = (r_sd, g_sd, b_sd)
    return mean, std

class Change_AMS():
    def __init__(self, csv_file, path, mean, sd, nojitter=False):
        self.dataset = pd.read_csv(csv_file)
        self.path = path
        self.mean = mean
        self.sd = sd
        self.nojitter = nojitter
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        names = ['anchor', 'positive', 'negative']
        triplet_labels = [row[x + '_dt'] for x in names]
        triplet_path = [os.path.join(self.path, row['nb'], row['cluster'], row[x + '_path']) for x in names]
        # if self.zfs:
        #     ## Try to open the images from zfs
        #     try:
        #         ## path default = ./
        #     triplet_path_zfs = [os.path.join('/ivi/xfs/', self.path, row['nb'], row['cluster'], row[x + '_path']) for x in names]
        #     ## If they don't exist, copy the files to zfs storage
        #     except FileNotFoundError:
        #         for path_hdd in triplet_path = [os.path.join(self.path, row['nb'], row['cluster'], row[x + '_path']) for x in names]:
        #             os.system('cp ')
        try:
            triplet_imgs = [Image.open(x) for x in triplet_path]
        except UnidentifiedImageError:
            if '/home/talpher/phd/sscd/antenna_420_126_test' in triplet_path[0]:
                triplet_path_og = [x.replace('antenna_420_126_test','antenna_4k_12') for x in triplet_path]
                triplet_imgs = [Image.open(x).resize((420,126)) for x in triplet_path_og]
                for img, path in zip(triplet_imgs, triplet_path):
                    img.save(path)
            else:
                triplet_path = [os.path.join(self.path.replace('/ivi/xfs/talpher/','./'), row['nb'], row['cluster'], row[x + '_path']) for x in names]
                triplet_imgs = [Image.open(x) for x in triplet_path]
        datetimes = [date.fromisoformat(x) for x in triplet_labels]
        triplet_distances = [datetimes[1] - datetimes[0], datetimes[2] - datetimes[1], datetimes[2] - datetimes[0]]
        triplet_distances = [abs(x.days) for x in triplet_distances]
        return [imgtotensor(x, self.mean, self.sd, nojitter=self.nojitter) for x in triplet_imgs], torch.tensor(triplet_distances)

def sum_params(model):
    t = 0
    for p in model.parameters():
        t += torch.sum(p)
    return t

def Tweaked_TripletMarginLoss(a,p,n, labels=False, margin=1):
    pdist = nn.PairwiseDistance(p=2)    
    pos_distance = pdist(a,p)
    neg_distance = pdist(a,n)

    diff = pos_distance - neg_distance +margin
    #print(diff)
    max_0 = torch.where(diff>0, diff, 0)
    #print(max_0)
    loss = torch.mean(max_0)
    acc = sum(max_0==0)/len(max_0)
    return loss, acc

def piecewise(x, p=365):
    a = 1/p

    return torch.where(x<p, 0.5*(a*x)**2, a*x-0.5)



def Tweaked_TripletMarginLoss_Linear(a,p,n, labels=False):

    anc_pos_delta, pos_neg_delta, anc_neg_delta = torch.tensor_split(labels, 3, dim=1)
    pdist = nn.PairwiseDistance(p=2)    
    pos_distance = pdist(a,p)
    neg_distance = pdist(a,n)
    #print('pos_distance:', pos_distance)
    #print('neg_distance:', neg_distance)
    #margin = anc_neg_delta.T[0]/anc_pos_delta.T[0] * pos_distance
    #print(pos_neg_delta)
    margin = piecewise(pos_neg_delta).squeeze(1)
    #print(margin)
    #sys.exit(0)
    pd_diff = pos_distance - neg_distance
    diff = pd_diff+margin
    #diff1 = pd_diff+1
    max_0 = torch.where(diff>0, diff, 0)
    #max_1 = torch.where(diff1>0, diff, 0)
    loss = torch.mean(max_0)
    acc = sum(max_0==0)/len(max_0)
    return loss, acc

def Tweaked_TripletMarginLoss_Adaptive(a,p,n, labels=False):

    anc_pos_delta, pos_neg_delta, anc_neg_delta = torch.tensor_split(labels, 3, dim=1)
    pdist = nn.PairwiseDistance(p=2)    
    pos_distance = pdist(a,p)
    neg_distance = pdist(a,n)
    #print('pos_distance:', pos_distance)
    #print('neg_distance:', neg_distance)
    margin = anc_neg_delta.T[0]/anc_pos_delta.T[0] * pos_distance
    pd_diff = pos_distance - neg_distance
    diff = pd_diff+margin
    diff1 = pd_diff+1
    max_0 = torch.where(diff>0, diff, 0)
    max_1 = torch.where(diff1>0, diff, 0)
    loss = torch.mean(max_0)
    acc = sum(max_1==0)/len(max_1)
    return loss, acc



def calculate_metrics(y_true, y_pred, extended=False):
    if not extended:
        # Calculate True Positives, False Positives, False Negatives
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))
        # Calculate Precision, Recall, and F1 Score
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        return precision, recall, f1_score
    else:
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        true_negatives = np.sum((y_true == 0) & (y_pred == 0))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))
        # Calculate Precision, Recall, and F1 Score
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        FPR = false_positives / (false_positives + true_negatives)

        return precision, recall, f1_score, FPR

def location_sampling(seq_clusters, s, split):
    location = seq_clusters.loc[seq_clusters['cluster'].str.contains(s)]['cluster'].unique()
    choice = np.random.choice(len(location), size=int(len(location)*split), replace=False)
    ind = np.zeros(len(location), dtype=bool)
    ind[choice] = True
    rest = ~ind
    train = list(location[ind])
    test = list(location[rest])
    return train, test

def perform_forward_2(model, im1, im2, headtype):
    if headtype == 'DINOV2':
        out1 = model.forward(im1, interpolate_pos_encoding=True)[1]
        out2 = model.forward(im2, interpolate_pos_encoding=True)[1]

    elif headtype == 'DINODINO' or headtype =='DINODINO14'or headtype =='resnet':
        out1 = model.forward(im1)
        out2 = model.forward(im2)

    elif headtype == 'clip':
        out1 = model.get_image_features(im1)
        out2 = model.get_image_features(im2)

    else:
        out1 = model.forward(im1, headtype)
        out2 = model.forward(im2, headtype)

    return out1, out2

def perform_forward_3(model, anc, pos, neg, headtype):

    if headtype == 'DINOV2':
        out_anc = model.forward(anc, interpolate_pos_encoding=True)[1]
        out_pos = model.forward(pos, interpolate_pos_encoding=True)[1]
        out_neg = model.forward(neg, interpolate_pos_encoding=True)[1]

    elif headtype == 'DINODINO' or headtype =='DINODINO14':
        out_anc = model.forward(anc)
        out_pos = model.forward(pos)
        out_neg = model.forward(neg)

    else:
        out_anc = model.forward(anc, headtype)
        out_pos = model.forward(pos, headtype)
        out_neg = model.forward(neg, headtype)

    return out_anc, out_pos, out_neg



def copy2xfs(df, input_dir, xfs_dir='/ivi/xfs/talpher/'):
    path_dir = {}
    regular_df= pd.read_csv(df)
    ## randomize rows
    df = regular_df.sample(frac=1)
    for i in range(len(df)):
        row = df.iloc[i]
        nb = row['nb']
        cluster = row['cluster']
        cluster_path = os.path.join(nb, cluster)

        old_dir = os.path.join(input_dir, cluster_path)
        xfs_rel_dir = os.path.join(xfs_dir, old_dir)
        for column in ['anchor_path', 'positive_path', 'negative_path']:
            img = row[column]
            old_path = os.path.join(old_dir, img)
            xfs_path = os.path.join(xfs_rel_dir, img)
            #print(old_path, xfs_path)
            if (xfs_path not in path_dir) and not os.path.isfile(xfs_path):
                path_dir[xfs_path] = 1
                os.makedirs(xfs_rel_dir, exist_ok=True)
                os.system('cp {} {}'.format(old_path, xfs_path))
        print('copied to xfs {}/{}'.format(i, len(df)))



def copy2zfs(df, input_dir, size, zfs_dir='/home/talpher/'):
    path_dir = {}
    regular_df= pd.read_csv(df)
    ## randomize rows
    df = regular_df.sample(frac=1)
    h, w = size
    print(h, w)
    assert h < w
    for i in range(len(df)):
        row = df.iloc[i]
        nb = row['nb']
        cluster = row['cluster']
        cluster_path = os.path.join(nb, cluster)

        old_dir = os.path.join('antenna_4k_12', cluster_path)
        new_dir = os.path.join(input_dir, cluster_path)
        zfs_rel_dir = os.path.join(zfs_dir, new_dir)
        for column in ['anchor_path', 'positive_path', 'negative_path']:
            img = row[column]
            old_path = os.path.join(old_dir, img)
            zfs_path = os.path.join(zfs_rel_dir, img)
            #print(old_path, xfs_path)
            if (zfs_path not in path_dir) and not os.path.isfile(zfs_path):
                path_dir[zfs_path] = 1
                os.makedirs(zfs_rel_dir, exist_ok=True)
                #os.system('cp {} {}'.format(old_path, xfs_path))

                img = Image.open(old_path)
                img = img.resize((w, h))
                img.save(zfs_path)
        print('copied to zfs {}/{}'.format(i, len(df)))

class Consumer(multiprocessing.Process):

    def __init__(self, task_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print('%s: Exiting' % proc_name)
                self.task_queue.task_done()
                break

            next_task()
            self.task_queue.task_done()
        return


def resize(row, input_dir, zfs_dir, size, skip_switch):
    nb = row['nb']
    cluster = row['cluster']
    cluster_path = os.path.join(nb, cluster)
    h, w = size
    old_dir = os.path.join('antenna_4k_12', cluster_path)
    new_dir = os.path.join(input_dir, cluster_path)
    zfs_rel_dir = os.path.join(zfs_dir, new_dir)
    try:
        for column in ['anchor_path', 'positive_path', 'negative_path']:
            img = row[column]
            old_path = os.path.join(old_dir, img)
            zfs_path = os.path.join(zfs_rel_dir, img)
            #print(old_path, xfs_path)
            if not os.path.isfile(zfs_path):
                os.makedirs(zfs_rel_dir, exist_ok=True)
                #os.system('cp {} {}'.format(old_path, xfs_path))

                img = Image.open(old_path)
                img = img.resize((w, h))
                img.save(zfs_path)
    except KeyError:
        for column in ['im1', 'im2']:
            img = row[column]
            old_path = os.path.join('antenna_4k_12', img)
            ## try the path of the actual dir
            potential_old_path = os.path.join(input_dir, img)
            zfs_rel_dir = os.path.join(zfs_dir, input_dir)
            zfs_path = os.path.join(zfs_rel_dir, img)
            #print(old_path, xfs_path)
            #print(old_path, zfs_path)

            if not os.path.isfile(zfs_path):
                os.makedirs(os.path.join(zfs_dir, new_dir), exist_ok=True)
                
                #os.system('cp {} {}'.format(old_path, xfs_path))
                try:
                    img = Image.open(potential_old_path)
                    print('not resized!')

                except FileNotFoundError:
                    #print(old_path)
                    img = Image.open(old_path)
                    img = img.resize((w, h))
                if not skip_switch:
                    img.save(zfs_path)

    

class Task(object):
    def __init__(self, row, input_dir, zfs_dir, size, skip_switch):
        #print('Task init')
        self.row = row
        self.input_dir = input_dir
        self.zfs_dir = zfs_dir
        self.size = size
        self.skip_switch = skip_switch
    def __call__(self):
        #print('Task call')
        resize(self.row, self.input_dir, self.zfs_dir, self.size, self.skip_switch)
        
        #seq_24(self.args, self.info)
        #print('Inside call', self.info)
        
    def __str__(self):
        return self.info['filename']



def copy2zfs_mp(df, input_dir, size, zfs_dir='/home/talpher/', skip_switch=False):

    # Establish communication queues
    tasks = multiprocessing.JoinableQueue()

    # Start consumers
    num_consumers = multiprocessing.cpu_count() * 2
    #num_consumers = 1
    print('Creating %d consumers' % num_consumers)
    consumers = [Consumer(tasks) for i in range(num_consumers)]
    for w in consumers:
        w.start()

    #### Give the list of rows to the Taskforce
    regular_df= pd.read_csv(df)
    df = regular_df.sample(frac=1)

    
    for i in range(len(df)):
        tasks.put(Task(df.iloc[i], input_dir, zfs_dir, size, skip_switch))

    # Add a poison pill for each consumer
    for i in range(num_consumers):
        tasks.put(None)

    pbar = tqdm(total=tasks.qsize())

    last_queue = tasks.qsize()

    while tasks.qsize() > 0:
        diff = last_queue - tasks.qsize()
        pbar.update(diff)
        last_queue = tasks.qsize()
        sleep(0.2)



    # Wait for all of the tasks to finish
    tasks.join()







