import pickle
from sklearn.neighbors import NearestNeighbors
import numpy as np
from PIL import Image
import os
import torchvision.transforms as transforms
import tqdm
import torch

PHOTO_ROOT = './dataset/photo_test'
SKETCH_ROOT = './dataset/sketch_test'

mAP_list =[]
for i in range(20,21):
    print('---------------',i,'---------------')
    photoft_root = 'features_pkl/rn50_bs32_mg1_lr3_10class/'+str(i)+'/photo-resnet-epoch_'+str(i)+'.pkl'
    # print(photoft_root)
    photo_data = pickle.load(open(photoft_root, 'rb'))
    sketch_data = pickle.load(open('features_pkl/rn50_bs32_mg1_lr3_10class/'+str(i)+'/sketch-resnet-epoch_'+str(i)+'.pkl', 'rb'))

    # print(photo_data['name'][0])
    photo_feature = photo_data['feature']
    photo_name = photo_data['name']

    sketch_feature = sketch_data['feature']
    sketch_name = sketch_data['name']
    nbrs = NearestNeighbors(n_neighbors=np.size(photo_feature, 0),
                            algorithm='brute', metric='euclidean').fit(photo_feature)

    s_len = np.size(sketch_feature, 0)
    picked = [0] * s_len

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    count = 0
    count_5 = 0
    K = 30
    div = 0
    total_queries = 0 
    total_ap = 0.0
    mAP = 0.0
    for ii, (query_sketch, query_name) in tqdm.tqdm(enumerate(zip(sketch_feature, sketch_name))):
        query_sketch = np.reshape(query_sketch, [1, np.shape(query_sketch)[0]])
        query_class, query_img = os.path.split(query_name)
        if (query_class == 'shoe') :
            distances, indices = nbrs.kneighbors(query_sketch)

            div += distances[0][1] - distances[0][0]

            # top K

            retrieved_correctly = False
            count_relevant = 0
            number_retrieve= 0
            total_precision = 0.0
            # print("query_class: " ,query_class)
            for i, indice in enumerate(indices[0][:K]):
                number_retrieve +=1
                retrievaled_name = photo_name[indice]
                retrievaled_class, retrievaled_filename = os.path.split(retrievaled_name)

                retrievaled_name = os.path.splitext(retrievaled_filename)[0]

                if retrievaled_class == query_class:
                    if query_img.find(retrievaled_name) != -1:
                        if i == 0:
                            count +=1
                            # retrieved_correctly = True
                        count_5 += 1
                        break
                        # print (total_precision)
                    count_relevant += 1
                    total_precision += count_relevant/ number_retrieve

            if count_relevant == 0:
                total_ap += 0.0
            else:
                total_ap += total_precision/count_relevant   
            # print(ap)
            total_queries += 1
        else:
            continue

    mAP = total_ap / total_queries
    mAP_list.append(mAP)
    # print(mAP)
    print(f'mAP : {mAP:.16f}', f'Average precision: {total_ap:.10f}')