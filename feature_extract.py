# from data import TripleDataLoader
from utils.extractor import Extractor
from models.vgg import vgg16
# from models.sketch_resnet import resnet50
import torch as t
from torch import nn
import os

# The script to extract sketches or photos' features using the trained model

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

train_set_root = 'dataset/sketch_train'
test_set_root = 'dataset/sketch_test'

train_photo_root = 'dataset/photo_train'
test_photo_root = 'dataset/photo_test'

# The trained model root for resnet
# model at epoch 85th
# SKETCH_RESNET = '/data1/zzl/model/caffe2torch/mixed_triplet_loss/sketch/sketch_resnet_85.pth'
# PHOTO_RESNET = '/data1/zzl/model/caffe2torch/mixed_triplet_loss/photo/photo_resnet_85.pth'

# The trained model root for vgg
# SKETCH_VGG = 'model/sketch_vgg16_401.pth'
PHOTO_VGG = 'model/bt32_001_1/photo_vgg16_19.pth'
a = PHOTO_VGG.split('_')
epoch =''+ a[-1].split('.')[0]
# FINE_TUNE_RESNET = '/data1/zzl/model/caffe2torch/fine_tune/model_270.pth'

device = 'cuda:1'

'''vgg'''
vgg = vgg16(pretrained=False)
vgg.classifier[6] = nn.Linear(in_features=4096, out_features=125, bias=True)
# print(vgg)
# vgg = t.load(PHOTO_VGG, map_location=t.device('cpu'))
# vgg.load_state_dict(checkpoint)
# vgg.classifier[0] = nn.Linear(in_features=512*7*7, out_features=4096, bias=True)
# vgg.classifier[6] = nn.Linear(in_features=4096, out_features=125, bias=True)
vgg.load_state_dict(t.load(PHOTO_VGG, map_location=t.device('cpu')))
# vgg.cuda()

ext = Extractor(vgg)
ext.reload_model(vgg)
# vgg.eval()

photo_feature = ext._extract_with_dataloader(test_photo_root, 'feature/bt32_001_1/photo-vgg' + '-%sepoch.pkl'%epoch)
# vgg= t.load(SKETCH_VGG)
# ext.reload_model(vgg)

sketch_feature = ext._extract_with_dataloader(test_set_root, 'feature/bt32_001_1/sketch-vgg' + '-%sepoch.pkl'%epoch)

'''resnet'''

# resnet = resnet50()
# resnet.fc = nn.Linear(in_features=2048, out_features=125)
# resnet.load_state_dict(t.load(PHOTO_RESNET, map_location=t.device('cpu')))
# # resnet.cuda()

# ext = Extractor(pretrained=False)
# ext.reload_model(resnet)

# photo_feature = ext.extract_with_dataloader(test_photo_root, 'photo-resnet-epoch.pkl')

# resnet.load_state_dict(t.load(SKETCH_RESNET, map_location=t.device('cpu')))
# ext.reload_model(resnet)

# sketch_feature = ext.extract_with_dataloader(test_set_root, 'sketch-resnet-epoch.pkl')

