from PIL import Image
import numpy as np
import torchvision as tv
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
from sklearn.neighbors import NearestNeighbors
import pickle
import os

class Retrieval():
    def __init__(self, model,model_name):
        self.model = model
        self.model_name = model_name
        self.transform = tv.transforms.Compose([
                # tv.transforms.CenterCrop(224),
                tv.transforms.Resize(224),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
        if(self.model_name =='resnet50'):
            '''resnet50'''
            self.photo = pickle.load(open('features_pkl/rn50_bs32_mg1_lr3_10class/train/20/photo-resnet-epoch_20.pkl', 'rb')) #train
        elif (self.model_name == 'vgg16'):
            '''vgg16'''
            self.photo = pickle.load(open('features_pkl/vgg16/photo-vgg-29epoch1.pkl', 'rb'))
        self.feat_photo = self.photo['feature']
        self.name_photo = self.photo['name']

    def extract(self, sketch_src):
        self.model.eval()
        sketch_src = self.transform(sketch_src)
        if(self.model_name =='resnet50'):
            sketch_src = sketch_src.unsqueeze(0) 
            print("Đang extract ảnh bằng resnet50")
        else:
            print("Đang extract bằng VGG16")
        out = self.model(sketch_src)
        i_feature = out
        feature=i_feature.detach().numpy()
        return feature
    
    def retrieval(self, path):
        sketch_src = Image.open(path).convert('RGB')
        feat_sketch = self.extract(sketch_src)
        nbrs = NearestNeighbors(n_neighbors=90,algorithm='brute', 
                        metric='euclidean').fit(self.feat_photo)
        if(self.model_name =='resnet50'):
            '''resnet50'''
            query_sketch = feat_sketch
        elif (self.model_name == 'vgg16'):
            '''vgg16'''
            query_sketch = np.reshape(feat_sketch, [1, np.shape(feat_sketch)[0]])
        distances, indices = nbrs.kneighbors(query_sketch)
        path = []
        retrieve_photo = indices[0]
        
        for i in retrieve_photo:
            img ={}
            retrievaled_name = self.name_photo[i]
            real_path='dataset/photo_train/'+retrievaled_name
            img['path']=real_path
            img['name']=str(retrievaled_name)
            path.append(img)
        real_list_set=[]
        
        for i in range(5):
            real_list = []
            for j in range(18):
                name = retrieve_photo[i * 18 + j]
                real_path = 'dataset/photo_train/' + str(name)
                real_list.append((real_path, name))
            real_list_set.append(real_list)
        return real_list_set, path