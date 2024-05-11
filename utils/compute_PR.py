import pickle
from sklearn.neighbors import NearestNeighbors
import numpy as np
from PIL import Image
import os
import torchvision.transforms as transforms


PHOTO_ROOT = 'dataset/photo_test'

photo_data = pickle.load(open('feature/bt32_001_1/photo-vgg-19epoch.pkl', 'rb'))
photo_feature = photo_data['feature']
photo_name = photo_data['name']

def compute_PR(query_name, retriev_photo):
    count_1 = 0
    count = 0

    pre = 0
    recall = []
    ground = 20
    query_split = query_name.split('/')
    query_class = query_split[0]

    # top K
    x = 0
    for i in retriev_photo:
        retrievaled_name = photo_name[i]
        print(retrievaled_name)
        retrievaled_class = retrievaled_name.split('/')[0]

        retrievaled_name = retrievaled_name.split('/')[1]
        retrievaled_name = retrievaled_name.split('.')[0]

        # if i == 0: print('retriev:',retrievaled_class)
        x += 1
        if retrievaled_class == query_class:
            if x == 1:
                count_1 += 1
            count += 1
            # print (str(count) + '/' + str(x))
            pre += count / x
            recall.append(float(count/ground))
    
    pre_1 = count_1 / 1
    if count == 0: AP = 0
    else: AP = pre / count
    # print(str(pre)+'/'+str(count))
    print('pre@1 :', pre_1, '   AP :', AP)
    return AP



