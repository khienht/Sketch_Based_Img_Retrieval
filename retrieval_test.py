import pickle
from sklearn.neighbors import NearestNeighbors
import numpy as np
from PIL import Image
import os
import faiss
import torchvision.transforms as transforms
from utils.compute_PR import compute_PR
import tqdm

PHOTO_ROOT = 'dataset/photo_test'
SKETCH_ROOT = 'dataset/sketch_test'

photo_data = pickle.load(open('feature/bt32_001_1/photo-vgg-19epoch.pkl', 'rb'))
sketch_data = pickle.load(open('feature/bt32_001_1/sketch-vgg-19epoch.pkl', 'rb'))
photo_feature = photo_data['feature']

print(len(photo_feature))
# print(photo_feature[0])


i = 0

photo_name = photo_data['name']

sketch_feature = sketch_data['feature']
sketch_name = sketch_data['name']
print(sketch_feature[0].shape)
print(len(sketch_feature))
nbrs = NearestNeighbors(n_neighbors=30,algorithm='brute', 
                        metric='euclidean').fit(photo_feature)

count = 0
count_5 = 0
K = 5

AP = 0

for ii, (query_sketch, query_name) in tqdm.tqdm(enumerate(zip(sketch_feature, sketch_name))):
    # print('shape', query_sketch.shape)
    query_sketch = np.reshape(query_sketch, [1, np.shape(query_sketch)[0]])
    query_split = query_name.split('/')
    query_class = query_split[0]
    query_img = query_split[1]

    distances, indices = nbrs.kneighbors(query_sketch)
    # if query_class == 'starfish':
    print('query:', query_name)
    AP += compute_PR(query_name, indices[0])
    count += 1

    # if ii ==14: break
print(count)
mAP = AP / count
print('mAP :', mAP)

