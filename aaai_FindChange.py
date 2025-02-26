import numpy as np
import os
from time import time
import argparse
import pickle
import pandas as pd
import random
from glob import glob
import random
from collections import OrderedDict
from PIL import Image, ImageFont, ImageDraw 
## ipython notebook 
import matplotlib.pyplot as plt
import datetime
import sys
#import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
from IPython.display import clear_output
import matplotlib.pyplot as plt

import random
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import paired_distances

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import scipy
import seaborn as sns
from math import sin, cos, sqrt, atan2, radians


## Utils

def find_isolated_change(cd, thresh_center, thresh_boundaries=-1):
    if thresh_boundaries<0:
        thresh_boundaries = thresh_center
    cd = np.where(cd < thresh_boundaries, 0, cd)
    cd = np.where(cd > thresh_center, 1, cd)
    binary_squares = np.where(cd > 1, 2, cd)
    conv = np.ones((3,3))*2
    conv[1,1] = 1
    binary_squares = np.vstack((binary_squares[:,-1], binary_squares.T, binary_squares[:,0])).T
    return np.where(scipy.signal.convolve2d(binary_squares, conv, mode='same')==1, 1, 0)[:, 1:-1]

class patches_from_path():
    def __init__(self, path, CF):
        self.path = path
        pathHD = path.split('/')
        self.date = path.split('/')[-1][:8]
        if CF.sequence_length == 200:
            pathHD[0] = '1_timemachine_full'
            self.pathHD = os.path.join(*pathHD)
            self.img = Image.open(path)
            self.imgHD = Image.open(self.pathHD)
            self.imgHD = Image.fromarray(np.asarray(np.vstack((np.array(self.imgHD), np.zeros((80, 4000,3))))).astype(np.uint8))
        else:
            pathHD[0] = 'antenna_4k_12'
            self.pathHD = os.path.join(*pathHD)
            self.img = Image.open(path)
            self.imgHD = Image.open(self.pathHD)        
        self.patchsize = CF.patchsize
        self.sequence_length = CF.sequence_length
        self.tokenh = CF.tokenh
        self.tokenw = CF.tokenw
    def get_patch(self, i):
        if self.sequence_length == 200:
            y = int(i/25)
            x = int(i%25)
            patch = np.array((0,0,16,16))
            patch[0] += x*16
            patch[2] += x*16
            patch[1] += y*16
            patch[3] += y*16
        else:
            y = int(i/50)
            x = int(i%50)
            patch = np.array((0,0,14,14))
            patch[0] += x*14
            patch[2] += x*14
            patch[1] += y*14
            patch[3] += y*14
        return self.img.crop(patch)
    def get_patchHD(self,i, savepath=''):
        if self.sequence_length == 200:
            y = int(i/25)
            x = int(i%25)
            patch = np.array((0,0,160,160))
            patch[0] += x*160
            patch[2] += x*160
            patch[1] += y*160
            patch[3] += y*160
        elif self.sequence_length == 750:
            y = int(i/self.tokenw)
            x = int(i%self.tokenw)
            patch = np.array((0,0,80,80))
            patch[0] += x*80
            patch[2] += x*80
            patch[1] += y*80
            patch[3] += y*80
        else:
            y = int(i/self.tokenw)
            x = int(i%self.tokenw)
            patch = np.array((0,0,133,133))
            patch[0] += x*133
            patch[2] += x*133
            patch[1] += y*133
            patch[3] += y*133

        cropped = self.imgHD.crop(patch)
        if savepath:
            cropped.save(savepath)
        return cropped
    def get_multipatchHD(self, ii):
        if self.sequence_length == 200:
            i = ii[0,0] % 200
            y = np.floor(i/25).astype(int)
            x = i%25
            
            patch = np.array((0,0,ii.shape[0]*160,ii.shape[1]*160))
            patch[0] += x*160
            patch[2] += x*160
            patch[1] += y*160
            patch[3] += y*160
        ## 700 210
        elif self.sequence_length == 750:
            i = ii[0,0] % self.sequence_length
            y = np.floor(i/self.tokenw).astype(int)
            x = i%self.tokenw
            
            # patch = np.array((0,0,ii.shape[0]*140,ii.shape[1]*140))
            # patch[0] += x*140
            # patch[2] += x*140
            # patch[1] += y*140
            # patch[3] += y*140
            patch = np.array((0,0,ii.shape[0]*80,ii.shape[1]*80))
            patch[0] += x*80
            patch[2] += x*80
            patch[1] += y*80
            patch[3] += y*80
        
        ## 400 126
        else:
            i = ii[0,0] % self.sequence_length
            y = np.floor(i/self.tokenw).astype(int)
            x = i%self.tokenw
            
            # patch = np.array((0,0,ii.shape[0]*140,ii.shape[1]*140))
            # patch[0] += x*140
            # patch[2] += x*140
            # patch[1] += y*140
            # patch[3] += y*140
            patch = np.array((0,0,ii.shape[0]*133,ii.shape[1]*133))
            patch[0] += x*133
            patch[2] += x*133
            patch[1] += y*133
            patch[3] += y*133
        cropped = self.imgHD.crop(patch)
        #self.imgHD.save('tmppatch.jpg') for debugging purposes
        return cropped

def find_adaptive_change(cd, thresh_center, thresh_boundaries=-1):    
    iso_change = np.zeros(cd.shape)
        
    cd = np.vstack((cd[:,-1], cd.T, cd[:,0])).T
    y, x = cd.shape
    w=0
    for i in range(y):
        for j in range(x):
            if j==0 or j==x-1:
                continue
            element = cd[i,j]
            if element < thresh_center:
                continue
            if i==0:
                window = cd[i: i+2, j-1:j+2]
            elif i<y-1:
                window = cd[i-1: i+2, j-1:j+2]
            else:
                window = cd[i-1: i+2, j-1:j+2]
            window2 = window - (element/thresh_boundaries)
            #window2 = window - thresh_boundaries
            if len(np.where(window2.flatten()>0)[0]) == 1:
                iso_change[i, j-1] = 1
    return iso_change.astype(np.uint8)

def paths_to_lon_lat(t0):
    
    l = t0.split('/')[2].split('_')
    
    l1 = l[1]
    lon = l1[:1] + '.' + l1[1:]
    l2 = l[3]
    lat = l2[:2] + '.' + l2[2:]
    try:
        assert lon[:2] == '4.'
        assert lat[:3] == '52.'
    except:
        assert lon[:2] == '5.'
    return lon, lat

def mod13(x):
    if x[4:6] == '13':
        return str(int(x[:4])+1) + '01' + x[6:8]
    return x

def get_lon_lat_from_filenames(x):
    s = x.split('.')
    lon = s[0][-1] +'.'+ s[1][:-3]
    lat = s[1][-2:] +'.'+ s[2]
    return float(lon), float(lat)
    # s = x.split('_4.')[-1].split('_')
    # lon = '4.' + s[0]
    # lat = s[1][:-4]
    # try:
    #     return float(lon), float(lat)
    # except ValueError:
    #     print(x, lon, lat)
def Kurts_Magic(lon1, lat1, lon2, lat2):
    R = 6373.0
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance


def check_constraints(CF, c1, img1_no, img2_no, img1_date, img2_date, years_apart, season, same_season, same_month, within_distance):
    if within_distance:
        fn1 = CF.dict[c1.indices[img1_no][0]][0]
        fn2 = CF.dict[c1.indices[img2_no][0]][0]
        lon1, lat1 = get_lon_lat_from_filenames(fn1)
        lon2, lat2 = get_lon_lat_from_filenames(fn2)
        #print(lon1, lat1)
        #print(lon2, lat2)
        d  = Kurts_Magic(lon1, lat1, lon2, lat2) *1000
        if d > within_distance:
            return False

    if season =='winter':
        d1 = int(img1_date[4:6])
        d2 = int(img2_date[4:6])
        m1 = d1 in [12,13,1,2]
        m2 = d2 in [12,13,1,2]
        if not (m1 and m2):
            return False
            #print(d1, d2, 'winter!')
    elif season =='spring':
        d1 = int(img1_date[4:6])
        d2 = int(img2_date[4:6])
        m1 = d1 in [3,4,5]
        m2 = d2 in [3,4,5]
        if not (m1 and m2):
            return False
    elif season =='summer':
        d1 = int(img1_date[4:6])
        d2 = int(img2_date[4:6])
        m1 = d1 in [6,7,8]
        m2 = d2 in [6,7,8]
        if not (m1 and m2):
            return False
    elif season =='autumn':
        d1 = int(img1_date[4:6])
        d2 = int(img2_date[4:6])
        m1 = d1 in [9,10,11]
        m2 = d2 in [9,10,11]
        if not (m1 and m2):
            return False
    if same_season:
        # ddict = {   1:'w',
        #     2:'w',
        #     3:'ws',
        #     4:'ws',
        #     5:'ws',
        #     6:'s',
        #     7:'s',
        #     8:'s',
        #     9:'a',
        #     10:'a',
        #     11:'a',
        #     12:'w',
        #     13:'w'}
        ddict = {   1:'w',
            2:'w',
            3:'s',
            4:'s',
            5:'s',
            6:'su',
            7:'su',
            8:'su',
            9:'a',
            10:'a',
            11:'a',
            12:'w',
            13:'w'}
        m1 = int(img1_date[4:6])# in [10,11,12,1,2,3,4]
        m2 = int(img2_date[4:6])# in [10,11,12,1,2,3,4]
        if ddict[m1] != ddict[m2]:
            return False
    if same_month:
        m1 = int(img1_date[4:6])
        m2 = int(img2_date[4:6])
        if m1 != m2:
            return False
    if years_apart:
        try:
            dd1 = datetime.datetime.strptime(img1_date, '%Y%m%d')
            dd2 = datetime.datetime.strptime(img2_date, '%Y%m%d')
        except ValueError:
            dd1 = datetime.datetime.strptime(mod13(img1_date), '%Y%m%d')
            dd2 = datetime.datetime.strptime(mod13(img2_date), '%Y%m%d')
        margin = datetime.timedelta(days=365)*years_apart
        if not dd2-dd1 > margin:
            return False
    return True

def Make_Save_Directory(self, season, same_season, window_size, years_apart,
                                                  threshold, thresh_boundaries, actual_neighbourhood,
                                                  isolated_window, adaptive, calc_linear_change,
                                                  underthresh, same_month, within_distance, one_result_per_cluster):
    CR_dir = 'ChangeResults'
    if self.sequence_length != 750:
        CR_dir += '_'+str(self.sequence_length)
    if season:
        parent_dir = CR_dir + '/{}/{}'.format(season, self.dir.split('/')[-1])
    else:
        if not same_season:
            parent_dir = CR_dir + '/{}/'.format(self.dir.split('/')[-1])
        elif same_season:
            parent_dir = CR_dir + '/same_season/{}/'.format(self.dir.split('/')[-1])
        if same_month:
            parent_dir = CR_dir + '/same_month/{}/'.format(self.dir.split('/')[-1])
    child_dir = '{}{}_{}{}{}/{}'.format(window_size[0] if not years_apart else 'ya{}_'.format(years_apart) + str(window_size[0]), 
                                    window_size[1] if not isolated_window else str(window_size[1]) + '_iso{}'.format('ad' if adaptive else '') + str(thresh_boundaries), 
                                    '{}{}_calc_linear{}'.format(threshold, underthresh, calc_linear_change) if calc_linear_change else threshold, 
                                    '_wd_{}'.format(within_distance) if within_distance else '','_orpc' if one_result_per_cluster else '', actual_neighbourhood)
    
    savedir = os.path.join(parent_dir, child_dir)
    
    return savedir
    

def stitch_and_save(self, img_obj0, img_obj1, p0, p1, path0, path1, window_size, font):
    if self.sequence_length == 200:
        result = Image.new('RGB', (window_size[0]*320, window_size[1]*170))
        result.paste(im=img_obj0.get_multipatchHD(p0), box=(0, 0))
        result.paste(im=img_obj1.get_multipatchHD(p1), box=(window_size[0]*160, 0))
        draw = ImageDraw.Draw(result) 
        draw.text((window_size[0]*60, window_size[0]*160), img_obj0.date, font=font, align='left')
        draw.text((window_size[0]*240, window_size[0]*160), img_obj1.date, font=font) 
    elif self.sequence_length == 750:
        result = Image.new('RGB', (window_size[0]*160, window_size[1]*85))
        result.paste(im=img_obj0.get_multipatchHD(p0), box=(0, 0))
        result.paste(im=img_obj1.get_multipatchHD(p1), box=(window_size[0]*80, 0))
        draw = ImageDraw.Draw(result) 
        draw.text((window_size[0]*(-10 + (80)/2), window_size[0]*80), img_obj0.date, font=font, align='left')
        draw.text((window_size[0]*(-10 + 80 + (80/2)), window_size[0]*80), img_obj1.date, font=font)         
    else:
        result = Image.new('RGB', (window_size[0]*280, window_size[1]*150))
        result.paste(im=img_obj0.get_multipatchHD(p0), box=(0, 0))
        result.paste(im=img_obj1.get_multipatchHD(p1), box=(window_size[0]*133, 0))
        draw = ImageDraw.Draw(result) 
        draw.text((window_size[0]*(-10 + (140)/2), window_size[0]*140), img_obj0.date, font=font, align='left')
        draw.text((window_size[0]*(-10 + 140 + (140/2)), window_size[0]*140), img_obj1.date, font=font) 
    return result

def detect_changing_signal(icd, length=4):
    signals = []
    
    if type(icd) == list:
        icd = np.array(icd)
    if length==4:
        try:
            for i in range(icd.shape[1]):
                signal = icd[:,i]
                for s in range(len(signal)-2):
                    segment = signal[s:s+3]
                    asc = np.array([0, 2, 0])
                    if (segment == asc).all():
                        signals.append((i, s))
        except IndexError:
            print(icd, icd.shape)
    elif length==3:
        for i in range(icd.shape[1]):
            signal = icd[:,i]
            for s in range(len(signal)-1):
                segment = signal[s:s+2]
                asc = np.array([0, 2])
                desc = np.array([2, 0])
                if (segment == asc).all() or (segment==desc).all():
                    signals.append((i, s))
    
    return signals
    

def display_x_change(CF, inp, ic_windows, ic_cd, window_size, length=4):
    
    
    idx, window_start = inp
    wi = ic_windows[window_start:window_start+length]
    cd = ic_cd[window_start:window_start+length-1]
    ps = [x.reshape(-1, window_size[0], window_size[1])[idx] for x in wi] 
    
    img_objs = []
    paths = []
    for p in ps:
        path, _ = CF.dict[p[0,0]]
        paths.append(path)
        img_objs.append(patches_from_path(path, CF))

    if CF.sequence_length == 200:
        result = Image.new('RGB', (window_size[0] * 160 * len(img_objs), window_size[1]*160))
        for j, (img_obj, p) in enumerate(zip(img_objs, ps)):
            result.paste(im=img_obj.get_multipatchHD(p), box=(j*window_size[0]*160,0))
    if CF.sequence_length == 750:
        result = Image.new('RGB', (window_size[0] * 80 * len(img_objs), window_size[1]*80))
        for j, (img_obj, p) in enumerate(zip(img_objs, ps)):
            result.paste(im=img_obj.get_multipatchHD(p), box=(j*window_size[0]*80,0))
    else:
        result = Image.new('RGB', (window_size[0] * 133 * len(img_objs), window_size[1]*133))
        for j, (img_obj, p) in enumerate(zip(img_objs, ps)):
            result.paste(im=img_obj.get_multipatchHD(p), box=(j*window_size[0]*133,0))
        
    details=(np.array(cd)[:,idx], paths)
    return result, details

    
class cluster():
    ### c = cluster_dict_value
    def __init__(self, c, CF):
        self.indices, self.order = c
        self.indices = self.indices.reshape(-1, CF.sequence_length)
        self.dates = [CF.dict[x[0]][0].split('/')[3][:8] for x in self.indices]
        self.CF = CF
    ### patch_idxes = [0,1]
    ### method can be 'paired' or 'norm'
    def calc_distances(self, patch_idxes, window_size=(1,1), method='paired'):
        patch1_idxes = self.indices[patch_idxes[0]]
        patch2_idxes = self.indices[patch_idxes[1]]
        
        if method=='paired':
            token_distances = paired_distances(self.CF.patches[patch1_idxes], self.CF.patches[patch2_idxes]).reshape(self.CF.tokenh, self.CF.tokenw)
            window_dim = window_size[0] * window_size[1]
            slide = np.lib.stride_tricks.sliding_window_view(token_distances, window_size)
            patch1_window_indices = np.lib.stride_tricks.sliding_window_view(patch1_idxes.reshape(self.CF.tokenh, self.CF.tokenw), window_size)
            patch2_window_indices = np.lib.stride_tricks.sliding_window_view(patch2_idxes.reshape(self.CF.tokenh, self.CF.tokenw), window_size)
            window_indices = (patch1_window_indices, patch2_window_indices)
            slide = slide.reshape(slide.shape[0], slide.shape[1], window_dim)

            return np.mean(slide, axis=2), window_indices

    def token_distances(self, patch_idxes, window_size=(1,1), method='paired'):
        patch1_idxes = self.indices[patch_idxes[0]]
        patch2_idxes = self.indices[patch_idxes[1]]
        
        token_distances = paired_distances(self.CF.patches[patch1_idxes], self.CF.patches[patch2_idxes]).reshape(self.CF.tokenh, self.CF.tokenw)
        return token_distances

    def calc_slide_from_tokens(self, token_distances, patch_idxes, window_size=(1,1)):
        patch1_idxes = self.indices[patch_idxes[0]]
        patch2_idxes = self.indices[patch_idxes[1]]
        window_dim = window_size[0] * window_size[1]
        slide = np.lib.stride_tricks.sliding_window_view(token_distances, window_size)
        patch1_window_indices = np.lib.stride_tricks.sliding_window_view(patch1_idxes.reshape(self.CF.tokenh, self.CF.tokenw), window_size)
        patch2_window_indices = np.lib.stride_tricks.sliding_window_view(patch2_idxes.reshape(self.CF.tokenh, self.CF.tokenw), window_size)
        window_indices = (patch1_window_indices, patch2_window_indices)
        slide = slide.reshape(slide.shape[0], slide.shape[1], window_dim)
        return np.mean(slide, axis=2), window_indices        


class ChangeFinding():
    def __init__(self, saved_model_dir, no_patches = 110000, feature_length=768, precision='float64'):
        
        #### Load Memmap and Patchdict
        self.dir = saved_model_dir
        self.file_obj = 'random_patches_{}_{}'.format(no_patches, 0 if feature_length==768 else feature_length)
        self.memmap_path = os.path.join(self.dir, self.file_obj + '.memmap')
        self.dict_path = os.path.join(self.dir, self.file_obj + '.pickle') 
        self.dict = pickle.load(open(self.dict_path, 'rb'))
        self.patches = np.memmap(self.memmap_path, dtype=precision, mode='r', 
                                 shape=(len(self.dict.keys()),feature_length))
        print(saved_model_dir)
        if ('210_700' in saved_model_dir) or ('112_350' in saved_model_dir):
            print('using 210 700 length')
            p = 14
            s = 750
            h, w = (15, 50)
        elif '126_420' in saved_model_dir:
            print('using 126_420 length')
            p = 14
            s = 270
            h, w = (9, 30)
        else:
            p = 16
            s = 200
            h, w = (8, 25)
        self.patchsize = p
        self.sequence_length = s
        self.tokenh = h
        self.tokenw = w
        
    def extract_cluster_dict(self):
        try:
            self.cluster_path = os.path.join(self.dir, self.file_obj + '_clusters.pickle')
            self.cluster_dict = pickle.load(open(self.cluster_path, 'rb')) 
            print('Loaded dict from disk!')
        except FileNotFoundError:
            print('No dict on disk, generating...')
            key_list = []
            self.cluster_dict={}
            total = len(self.dict.keys())
            prev_img = ''
            img_tokens = []
            for key in range(len(self.dict.keys())):
                value = self.dict[key]
                current_cluster = ''.join(x + '/' for x in value[0].split('/')[:-1])
                current_img = value[0].split('/')[-1]
                if key == 0:
                    prev_cluster = ''.join(x + '/' for x in value[0].split('/')[:-1])
                    prev_img = ''.join(value[0].split('/')[-1])
                    img_list = [prev_img]
                if current_cluster == prev_cluster:
                    
                    if current_img == prev_img:
                        img_tokens.append(key)
                    else:
                        img_list.append(current_img)
                        prev_img = current_img
                        key_list.append(img_tokens)
                        img_tokens=[key]

                else:
                    key_list.append(img_tokens)
                    chronological_order = np.argsort(img_list)
                    #print(key_list, chronological_order)
                    key_list = np.array(key_list)[chronological_order]
                    self.cluster_dict[prev_cluster] = (key_list.flatten(), chronological_order)
                    prev_cluster = current_cluster
                    prev_img = current_img
                    key_list   = []
                    img_tokens=[key]
                    img_list   = [current_img]
                if key % 100000 == 0:
                    clear_output(wait=True)
                    print(key, '/', total)
                
            with open(self.cluster_path, 'wb') as f:
                pickle.dump(self.cluster_dict, f)
    
    def Visualize_Slice(self, cluster, token):
        k = list(self.cluster_dict.keys())[cluster]
        v = self.cluster_dict[k]
        inds, order = v
        print(order)
        cslice = self.patches[inds] 
        intervals = np.array([x for x in inds if x%sequence_length==0])
        indices = intervals + token
        #indices = indices
        rslice = cslice.reshape(-1, self.sequence_length, 768)
        rslice[:,token,:].shape
        ipatches=[]
        for i in indices:
            path, idx = self.dict[i]
            img_obj = patches_from_path(path,cluster.CF.patchsize)
            display(img_obj.get_patchHD(idx))
            ipatches.append(np.array(CF.patches[i]))
        for i in range(len(ipatches)-1):
            print(pairwise_distances(ipatches[i].reshape(1,-1), ipatches[i+1].reshape(1,-1)))
        return ipatches

    def Find_Change(self, window_size, threshold, neighbourhood='',
                    no_of_results=1000, same_season=False, season='',
                    calc_dist=False, no_save=False, isolated_window=False, thresh_boundaries=-1, save_idx=False,
                    years_apart=0, calc_linear_change=False, build_df=False, adaptive_boundary=False, underthresh=False, same_month=False, 
                   one_result_per_image_pair=False, build_comparison_dict=False, optimize_with_disk=False, build_disk_optimization=False, one_result_per_cluster=False,
                   within_distance=False):
        done=False
        comparisons=0
        comparison_dict = {}
        ### This section of code builds the precalculated_distances for a backbone so we only have to do the sliding window
        if build_disk_optimization:
            #### use a dict to keep track of the result idx + imagepairs -> idx in the array
            optimization_tracking_dict = {}
            precalculated_distances = np.memmap(os.path.join(self.dir, 'precalculated_distances.memmap'), dtype='float32', mode='w+', 
                                 shape=(203930,self.tokenh, self.tokenw))
            for result_idx, (key, value) in enumerate(self.cluster_dict.items()):
                c1 = cluster(value, self)
                for img1_no, img1_date in zip(range(len(c1.order)), c1.dates):
                    if done:
                        break
                    for img2_no, img2_date in zip(range(len(c1.order)), c1.dates):
                        if done:
                            break
                        if img2_no <= img1_no:
                            continue
                        if img1_no == img2_no:
                            continue
                        optimization_tracking_dict[(result_idx, img1_no, img2_no)] = comparisons
                        precalculated_distances[comparisons] = c1.token_distances([img1_no, img2_no], window_size=window_size, method='paired')
                        comparisons+=1
                        if comparisons % 1000 == 0 and comparisons!=0:
                            print('{}/{}'.format(comparisons, 203930))
                            
                        #cd, wi = c1.calc_distances([img1_no, img2_no], window_size=window_size, method='paired')
            print(comparisons)
            with open(os.path.join(self.dir, 'precalculated_distances_trackingdict.pickle'), 'wb') as writing_obj:
                pickle.dump(optimization_tracking_dict, writing_obj)
            sys.exit(0)
        if optimize_with_disk:
            optimization_tracking_dict = pickle.load(open(os.path.join(self.dir, 'precalculated_distances_trackingdict.pickle'), 'rb'))
            precalculated_distances = np.memmap(os.path.join(self.dir, 'precalculated_distances.memmap'), dtype='float32', mode='r', 
                                 shape=(203930,self.tokenh, self.tokenw))
        if calc_linear_change:
            deets_dict = {}
            df_dict = {}
        if build_df:
            if not args.calc_linear_change:
                location_df = pd.DataFrame(columns=['nb', 'lon', 'lat', 'idx', 'path0', 'path1'])
                loc_idx=0
            else:
                location_df = pd.DataFrame(columns=['nb', 'path'])
                loc_idx=0

        if save_idx:
            idxes_for_dbscan=[]
        if season:
            same_season=False
        if no_save:
            print('Not saving...')
        if calc_dist:
            alldistslist=[]
            percentile_list = []
            weight_list = []
            ### Calculates the distribution by taking the average of thresholds (top 50%, 25%, 10%, 5%, 1%)
            percentiles = np.array([.5, .25, .1, .05, 0.01])
        font = ImageFont.truetype(r'/home/talpher/phd/arial.ttf', 6*window_size[1]) 
        saved_results=0
        clusters_total = len(list(self.cluster_dict.items()))
        faulty_clusters = pickle.load(open('faulty_clusters.pickle', 'rb'))
        start = time()
        for result_idx, (key, value) in enumerate(self.cluster_dict.items()):
            if key in faulty_clusters:
                continue
            if done:
                break
            if no_save and build_df:
                if result_idx%100==0:
                    clear_output(wait=True)
                    print('{}/{}, results: {} out of {} in {} seconds'.format(result_idx, clusters_total, len(location_df), no_of_results, int(time()-start)))
                    start = time()
            elif not no_save or build_df:
                if result_idx%100==0:
                    clear_output(wait=True)
                    print('{}/{}, results: {} out of {} in {} seconds'.format(result_idx, clusters_total, saved_results, no_of_results, int(time()-start)))
                    start = time()
            else:
                if result_idx%100==0:
                    print(result_idx, clusters_total, int(time()-start))
                    start = time()
            actual_neighbourhood = key.split('/')[1]
            if neighbourhood:
                if actual_neighbourhood != neighbourhood:
                    continue
            c1 = cluster(value, self)
            if calc_dist:
                intra_cluster = []
            ## Double Change
            if calc_linear_change:
                if not actual_neighbourhood in df_dict.keys():
                    df_dict[actual_neighbourhood] = {'signals':0, 'comparisons':0}
                isolated_changes=[]
                intra_cluster_distances = []
                intra_cluster_windows = []
                intra_cluster_bools = []
                too_many_constraints=True
                for i in range(len(c1.order)-1):
                    combination = [i,i+1]
                    comparisons+=1
                    img1_no, img2_no = combination

                    if not check_constraints(self, c1, img1_no, img2_no, c1.dates[img1_no], c1.dates[img2_no], years_apart, season, same_season, same_month, within_distance):
                        continue

                    if optimize_with_disk: 
                        precalculated_distances_idx = optimization_tracking_dict[(result_idx, img1_no, img2_no)]
                        token_distances = precalculated_distances[precalculated_distances_idx]
                        
                        cd, wi = c1.calc_slide_from_tokens(token_distances, [img1_no, img2_no], window_size=window_size)

                    else:
                        cd, wi = c1.calc_distances(combination, window_size=window_size, method='paired')

                    if isolated_window:
                        if adaptive_boundary:
                            isolated_change = find_adaptive_change(cd, threshold, thresh_boundaries)
                        elif not adaptive_boundary:
                            isolated_change = find_isolated_change(cd, threshold, thresh_boundaries)
                        isolated_changes+= [y for y in np.where(isolated_change.flatten())[0]]

                    intra_cluster_distances.append(cd.flatten())

                    cd2 = np.where(cd.flatten()<underthresh, 0, cd.flatten())
                    cd2 = np.where(cd2>threshold, 2, cd2)
                    cd2 = np.where(cd2>2, 1, cd2)

                    intra_cluster_bools.append(cd2)
                    intra_cluster_windows.append(wi[0])
                    too_many_constraints=False
                if too_many_constraints:
                    continue
                intra_cluster_windows.append(wi[1])
                signals = detect_changing_signal(intra_cluster_bools, length=calc_linear_change)
                df_dict[actual_neighbourhood]['comparisons'] += comparisons
                for calc_linear_idx, signal in enumerate(signals):
                    if isolated_window:
                        if not signal[0] in isolated_changes:
                            continue

                    savedir = Make_Save_Directory(self, season, same_season, window_size, years_apart,
                          threshold, thresh_boundaries, actual_neighbourhood,
                          isolated_window, adaptive_boundary, calc_linear_change, underthresh, same_month, within_distance, one_result_per_cluster)

                    os.makedirs(savedir, exist_ok=True)
                    img, deets = display_x_change(self, signal, intra_cluster_windows, 
                                                  intra_cluster_distances,  window_size, length=calc_linear_change)

                    pathandkey = ''.join(x + '_' for x in [x.split('/')[-1][:8]+ '_pid_' + x.split('o_')[1][:11] for x in deets[1]])[:-1] + '_{}'.format(calc_linear_idx) + '.jpg'
                    deets_dict[pathandkey] = {}
                    deets_dict[pathandkey]['deets'] = deets
                    deets_dict[pathandkey]['signal'] = signal
                    df_dict[actual_neighbourhood]['signals'] +=1
                    
                    if build_df:
                        ### Get lon lat of cluster
                        #lon, lat = paths_to_lon_lat(path0)
                        row = [actual_neighbourhood, pathandkey]
                        location_df.loc[loc_idx] = row
                        loc_idx +=1



                    if not no_save:
                        img.save(os.path.join(savedir, pathandkey))
                    saved_results +=1
                    if saved_results >= no_of_results:
                        done=True
                    if one_result_per_image_pair:
                        break
                    
                continue
            cluster_done=False
            ## Loop over all imgs twice to do a pairwise comparison. Skip if they are the same image.
            for img1_no, img1_date in zip(range(len(c1.order)), c1.dates):
                if done or cluster_done:
                    break
                for img2_no, img2_date in zip(range(len(c1.order)), c1.dates):
                    if done or cluster_done:
                        break
                    if img2_no <= img1_no:
                        continue
                    if img1_no == img2_no:
                        continue
                    comparisons+=1

                    if not check_constraints(self, c1, img1_no, img2_no, img1_date, img2_date, years_apart, season, same_season, same_month, within_distance):
                        continue

                    if build_comparison_dict:
                        try:
                            comparison_dict[actual_neighbourhood] += 1
                        except KeyError:
                            comparison_dict[actual_neighbourhood] = 1
                        #continue
                    ####
                    if optimize_with_disk: 
                        precalculated_distances_idx = optimization_tracking_dict[(result_idx, img1_no, img2_no)]
                        token_distances = precalculated_distances[precalculated_distances_idx]
                        
                        cd, wi = c1.calc_slide_from_tokens(token_distances, [img1_no, img2_no], window_size=window_size)

                    else:
                        cd, wi = c1.calc_distances([img1_no, img2_no], window_size=window_size, method='paired')
                    #print(cd)
                    #print('calculated_distances')
                    ### loop over all windowed patches we have now.
                    ### Patch is the numerical value (feature) representing change for a given patch.
                    ### It is the mean difference of all the tokens in the window. 
                    
                    ### Check whether cd[idx-25] > threshold, cd[idx-1], cd[idx+1], cd[idx+25] 
                    ### (plus shape around the patch)
                    ### if these are under the threshold you can save it. 

                    if calc_dist:
                        intra_cluster.append(cd)
                    if no_save and not build_df:
                        continue
                   
                    ## If isolated_window we look for changes that do not have any neighbouring changes
                    ## This is done by passing a convolution window over the image [[2,2,2],[2,1,2],[2,2,2]]
                    ## All the values that are 1 are then considered isolated.
                    if isolated_window:
                        if adaptive_boundary:
                            isolated_change = find_adaptive_change(cd, threshold, thresh_boundaries)
                        elif not adaptive_boundary:
                            isolated_change = find_isolated_change(cd, threshold, thresh_boundaries)
                        
                        selected_idxes = np.where(isolated_change.flatten())[0]
                    
                    elif not isolated_window:
                            selected_idxes = np.where(cd.flatten()>threshold)[0]

                    for idx in selected_idxes:
                        if done:
                            break
                        patch = cd.flatten()[idx]

                        p0 = wi[0].reshape(-1, window_size[0], window_size[1])[idx]
                        p1 = wi[1].reshape(-1, window_size[0], window_size[1])[idx]

                        path0, idx0 = self.dict[p0[0,0]]
                        path1, idx1 = self.dict[p1[0,0]]

                        if build_df:
                            ### Get lon lat of cluster
                            lon, lat = paths_to_lon_lat(path0)
                            row = [actual_neighbourhood, lon, lat, idx0, path0, path1]
                            location_df.loc[loc_idx] = row
                            loc_idx +=1
                        if no_save:
                            ### this no save is to make sure we can build the dataframe but continue without saving
                            pass
                        else:
                            img_obj0 = patches_from_path(path0, self)
                            img_obj1 = patches_from_path(path1, self)
                                
                            
                            if save_idx:
                                #Save idxes only implemented for windows size 11 because we cant really cluster tokens of larger sizes unless we concatenate
                                for x in p0.flatten():
                                    idxes_for_dbscan.append(int(x))
                                for x in p1.flatten():
                                    idxes_for_dbscan.append(int(x))
                            ### Makedir when we're sure we've found a same season patch

                            savedir = Make_Save_Directory(self, season, same_season, window_size, years_apart,
                                                          threshold, thresh_boundaries, actual_neighbourhood,
                                                          isolated_window, adaptive_boundary, calc_linear_change, underthresh, same_month, within_distance, one_result_per_cluster)

                            os.makedirs(savedir, exist_ok=True)

                            result = stitch_and_save(self, img_obj0, img_obj1, p0, p1, path0, path1, window_size, font)

                            ## specified font size
                            path0 = path0.split('/')[-1]
                            path1 = path1.split('/')[-1]
                            fpath0 = path0[:8] + '_pid_' + path0.split('_')[2]
                            fpath1 = path1[:8] + '_pid_' + path1.split('_')[2]
                            filename = '{}_{}_{}_{}.jpg'.format(fpath0,
                                                             fpath1,
                                                             idx, 
                                                             str(patch)[:6].replace('.', '-'))
                            savepath = os.path.join(savedir, filename)
                            result.save(savepath)
                            saved_results +=1
                        #display(result)
                        if one_result_per_cluster:
                            cluster_done=True
                            break                        
                        if saved_results >= no_of_results:
                            done=True
                        if one_result_per_image_pair:
                            break


            if calc_dist and (len(intra_cluster) > 0):
                #weight = len(intra_cluster)
                #intra_cluster = np.array(intra_cluster).flatten()
                #sorted_c1 = np.sort(intra_cluster)
                #elbo = [min(sorted_c1[-int(len(sorted_c1) * p):]) for p in percentiles]
                
                #percentile_list.append(elbo)
                #weight_list.append(weight)
                
                alldistslist += list(np.round(intra_cluster, 5))
        # if optimize_with_disk:# and not precalculated_comparisons_dict_loaded:
        #     with open(os.path.join(self.dir, 'precalculated_comparisons_dict.pickle'), 'wb') as writing_obj:
        #         pickle.dump(precalculated_comparisons_dict, writing_obj, protocol = pickle.HIGHEST_PROTOCOL)
        if save_idx:
            print(os.path.join(''.join(x +'/'for x in savedir.split('/')[:-1]), 'dbscan_idx.pickle'))
            with open(os.path.join(''.join(x +'/'for x in savedir.split('/')[:-1]), 'dbscan_idx.pickle'), 'wb') as z:
                pickle.dump(idxes_for_dbscan, z)
        if calc_dist:
            #if season:
            #    calc_dist_path = 'ChangeResults/{}/{}/'.format(season, self.dir.split('/')[-1])
            #else:
            #    if not same_season:
            #        calc_dist_path = 'ChangeResults/{}/'.format(self.dir.split('/')[-1])
            #    elif same_season:
            #        calc_dist_path = 'ChangeResults/same_season/{}/'.format(self.dir.split('/')[-1])
            
            calc_dist_path = Make_Save_Directory(self, season, same_season, window_size, years_apart,
                                          threshold, thresh_boundaries, actual_neighbourhood,
                                          isolated_window, adaptive_boundary, calc_linear_change, underthresh, same_month, within_distance, one_result_per_cluster)
            calc_dist_path = ''.join(x +'/' for x in calc_dist_path.split('/')[:-1])
            print(calc_dist_path)
            os.makedirs(calc_dist_path, exist_ok=True)
            
            os.makedirs(calc_dist_path, exist_ok=True)
            d = percentile_list
            w = weight_list

            m = np.array(d).T * np.array(w)
            thresholds = np.sum(m.T, axis=0)/np.sum(w)
            #with open(os.path.join(calc_dist_path, 'calculated_distances_{}{}.pickle'.format(window_size[0],
            #                                                                                 window_size[1] if not isolated_window else str(window_size[1]) + '_iso'+ str(thresh_boundaries))), 'wb') as open_file:
            #    pickle.dump((percentiles, thresholds, d, w, all_distslist), open_file)
            with open(os.path.join(calc_dist_path, 'calculated_distances_{}{}.pickle'.format(window_size[0],
                                                                                             window_size[1] if not isolated_window else str(window_size[1]) + '_iso'+ str(thresh_boundaries))), 'wb') as open_file:
                 pickle.dump((alldistslist), open_file)
            
            
            print('Thresholds:', thresholds)
        # Total amount of comparisons in our entire set
        if no_save and build_df:
            saved_results = len(location_df)
        analysis = 'We did {} comparisons and {} saved {} results. This threshold was the top {}%'.format(comparisons, 'havent' if no_save else '' ,saved_results, saved_results/comparisons if comparisons !=0 else 0)
        print(analysis)
        savedir = Make_Save_Directory(self, season, same_season, window_size, years_apart,
                  threshold, thresh_boundaries, actual_neighbourhood,
                  isolated_window, adaptive_boundary, calc_linear_change, underthresh, same_month, within_distance, one_result_per_cluster)
        savedir = ''.join(x +'/' for x in savedir.split('/')[:-1])
        
        os.makedirs(savedir, exist_ok=True)
        with open(os.path.join(savedir, 'analysis.pickle'), 'wb') as a:
            pickle.dump(analysis, a)
        if build_comparison_dict:
            with open(os.path.join(savedir, 'comparisons.pickle'), 'wb') as a:
                pickle.dump(comparison_dict, a)
        if build_df:            
            df_path = ''.join(y + '/' for y in savedir.split('/')[:-1])[:-1] + '.csv'
            
            location_df.to_csv(df_path)
            print('Saved df to', df_path)
        if calc_linear_change:
            with open(os.path.join(savedir, 'deetsdict.pickle'), 'wb') as z:
                pickle.dump(deets_dict, z)
            with open(os.path.join(savedir, 'df_dict.pickle'), 'wb') as z:
                pickle.dump(df_dict, z)
        print('Saved to', savedir)

        

    def VisualizeHeatmap(self, c, indices, heatmaptype='individual', resolution='original', norm=False, frac=0.12, pad=0.04, ticksize=10):
        ## Visualizes a heatmap for two images (cluster and indices)
        ## Heatmaptype individual gives us a heatmap only calibrated for a single pair. 
        ## This means we do not do normalizing over the entire testset.
        value = self.cluster_dict[c]
        c1 = cluster(value, self)
        cd, wi = c1.calc_distances(indices, method='paired')

        if heatmaptype=='individual':
            if resolution == 'HD':
                display(Image.open(self.dict[c1.indices[indices[0]][0]][0].replace('antenna_400_128_test', '1_timemachine_full')).crop((0,0,4000,1280)))
                display(Image.open(self.dict[c1.indices[indices[0]][0]][0].replace('antenna_400_128_test', '1_timemachine_full')).crop((0,0,4000,1280)))
            else:
                display(Image.open(self.dict[c1.indices[indices[0]][0]][0]))
                display(Image.open(self.dict[c1.indices[indices[1]][0]][0]))
            plt.figure(figsize=(20,6.4))
            plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False, labelsize=ticksize)
            if norm:
                plt.imshow(cd/norm, vmax=1, vmin=0)
            else:
                plt.imshow(cd)
            plt.tight_layout()
            plt.colorbar(location='bottom', fraction=frac, pad=0.04)
            plt.cb.ax.tick_params(labelsize=font_size)









def main(args):
    np.random.seed(0)

    ## change this to backbones
    CF = ChangeFinding(args.pretrained, no_patches = 109763, precision ='float32')

    CF.extract_cluster_dict()
    CF.Find_Change(args.window, args.threshold, 
                    neighbourhood=args.neighbourhood,
                    no_of_results=args.no_of_results,
                    same_season = args.same_season,
                    season=args.season, 
                    calc_dist=args.calc_dist,
                    no_save=args.no_save,
                    isolated_window =args.isolated_window,
                    thresh_boundaries=args.thresh_boundaries,
                    save_idx=args.save_indices_for_dbscan,
                    years_apart=args.years_apart,
                    build_df=args.build_df,
                    calc_linear_change=args.calc_linear_change,
                    adaptive_boundary=args.adaptive_boundary,
                    underthresh=args.underthresh,
                    same_month=args.same_month,
                    one_result_per_image_pair=args.one_result_per_image_pair,
                    build_comparison_dict=args.build_comparison_dict,
                    optimize_with_disk=args.optimize_with_disk,
                    build_disk_optimization = args.build_disk_optimization,
                    one_result_per_cluster = args.one_result_per_cluster,
                    within_distance = args.within_distance)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--window', nargs="+", type=int, default=[1,1])
    parser.add_argument('--threshold', type=float, default=24)
    parser.add_argument('--neighbourhood', type=str, default='')
    parser.add_argument('--pretrained', type=str, default='', required=True)
    parser.add_argument('--no_of_results', type=int, default=10)
    parser.add_argument('--calc_dist', type=int, default=0)
    parser.add_argument('--same_season', type=int, default=1)
    parser.add_argument('--season', type=str, default='')
    parser.add_argument('--isolated_window', type=int, default=0)
    parser.add_argument('--thresh_boundaries', type=float, default=-1)
    parser.add_argument('--no_save', type=int, default=0)
    parser.add_argument('--save_indices_for_dbscan', type=int, default=0)
    parser.add_argument('--calc_linear_change', type=int, default=0)
    parser.add_argument('--build_df', type=int, default=0)
    parser.add_argument('--adaptive_boundary', type=int, default=0)
    parser.add_argument('--underthresh', type=int, default=0)
    parser.add_argument('--years_apart', type=int, default=0)
    parser.add_argument('--same_month', type=int, default=0)
    parser.add_argument('--one_result_per_image_pair', type=int, default=0)
    parser.add_argument('--build_comparison_dict', type=int, default=0)
    parser.add_argument('--optimize_with_disk', type=int, default=0)
    parser.add_argument('--build_disk_optimization', type=int, default=0)
    parser.add_argument('--one_result_per_cluster', type=int, default=0)
    parser.add_argument('--within_distance', type=float, default=0)

    args = parser.parse_args()
    
    main(args)
