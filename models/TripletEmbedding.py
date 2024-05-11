import torch as t
from torch import nn
from data.triplet_input import TripleDataset
from data import TripleDataLoader
# import torch.utils.data as dataloader
from torch.utils.data import DataLoader 
from models.vgg import vgg16
from models.sketch_resnet import resnet50
# from utils.visualize import Visualizer
from torchnet.meter import AverageValueMeter
# import tqdm
from utils.extractor import Extractor
from sklearn.neighbors import NearestNeighbors
from torch.nn import DataParallel
from .TripletLoss import TripletLoss
from utils.test import Tester
import os
import numpy as np

class Config(object):
    def __init__(self):
        return

class TripletNet(object):

    def __init__(self, opt):
        # train config
        self.photo_root = opt.photo_root
        self.sketch_root = opt.sketch_root
        self.batch_size = opt.batch_size
        # self.device = opt.device
        self.epochs = opt.epochs
        self.lr = opt.lr
        self.log_interval = opt.log_interval

        # testing config
        self.photo_test = opt.photo_test
        self.sketch_test = opt.sketch_test
        self.test = opt.test
        self.test_f = opt.test_f

        self.save_model = opt.save_model
        self.save_dir = opt.save_dir

        # vis
        # self.vis = opt.vis
        # self.env = opt.env

        # fine_tune
        # self.fine_tune = opt.fine_tune
        # self.model_root = opt.model_root


        # dataloader config
        data_opt = Config()
        data_opt.photo_root = opt.photo_root
        data_opt.sketch_root = opt.sketch_root
        data_opt.batch_size = opt.batch_size

        self.dataloader_opt = data_opt

        # triplet config
        self.margin = opt.margin
        self.p = opt.p

        # feature extractor net
        self.net = opt.net
        self.cat = opt.cat

    def _get_vgg16(self, pretrained=True):
        model = vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=125, bias=True)
        return model

    def _get_resnet50(self, pretrained=True):
        model = resnet50(pretrained=pretrained)
        model.fc = nn.Linear(in_features=2048, out_features=125)

        return model

    def train(self):
        if self.net == 'vgg16':
            photo_net = self._get_vgg16()
            sketch_net = self._get_vgg16()
        elif self.net == 'resnet50':
            photo_net = self._get_resnet50()
            sketch_net = self._get_resnet50()

        
        print('net')
        print(photo_net)

        triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=self.p)

        # optimizer
        photo_optimizer = t.optim.Adam(photo_net.parameters(), lr=self.lr)
        sketch_optimizer = t.optim.Adam(sketch_net.parameters(), lr=self.lr)

        data_loader = TripleDataLoader(self.dataloader_opt)
        dataset = data_loader.load_data()
        print('Len:', len(dataset))
        for ii, data in enumerate(dataset):
            if ii==2: break
            print(data['L'])

        for epoch in range(self.epochs):

            print('---------------{0}---------------'.format(epoch))

            photo_net.train()
            sketch_net.train()
            avg_loss = 0
            text = []

            for ii, data in enumerate(dataset):
                photo_optimizer.zero_grad()
                sketch_optimizer.zero_grad()

                anchor = data['A']
                anchor = anchor.unsqueeze(0)
                pos = data['P']
                pos = pos.unsqueeze(0)
                neg = data['N']
                neg = neg.unsqueeze(0)
                label = data['L']
                label1 = data['L1']

                # label = t.unsqueeze(t.tensor(label), 0)
                
                _, a_feature = sketch_net(anchor)
                _, p_feature= photo_net(pos)
                _, n_feature= photo_net(neg)

                # a_feature, p_feature,n_feature = photo_net(anchor, pos, neg)

                loss = triplet_loss(a_feature, p_feature, n_feature)
                # loss = loss / self.batch_size

                loss.backward()

                # update param for model
                photo_optimizer.step()
                sketch_optimizer.step()

                print('[Train] Epoch: [{0}][{1}/{2}]\t'
                        'Triplet loss  ({triplet_loss_meterr:.3f})\t'
                        # 'Loss  ({losss:.4f})\t'
                        # 'Sketch Loss ({acc:.4f})\t'
                        .format(epoch + 1, ii + 1, len(dataset), triplet_loss_meterr=loss.item()))
                avg_loss += loss.item()
                text.append('[Train] Epoch: [{0}][{1}/{2}]\t'
                        'Triplet loss  ({triplet_loss_meterr:.3f})\t'
                        .format(epoch + 1, ii + 1, len(dataset), triplet_loss_meterr=loss.item()))
                if ii==3: break
            if self.save_model:
                t.save(photo_net.state_dict(), self.save_dir + '/photo' + '/photo_' + self.net + '_%s.pth' % epoch)
                t.save(sketch_net.state_dict(), self.save_dir + '/sketch' + '/sketch_' + self.net + '_%s.pth' % epoch)
                # t.save(photo_net, self.save_dir + '/photo' + '/photo_' + self.net + '_%s.pth' % epoch)
                # t.save(sketch_net, self.save_dir + '/sketch' + '/sketch_' + self.net + '_%s.pth' % epoch)
                with open('loss.txt', 'a') as file:
                    # Viết nội dung vào file
                    file.write("---------------{0}---------------\n".format(epoch))
                    for x in text:
                        file.write(x+".\n")
                    file.write("Train loss: " + str(avg_loss/len(dataset)) + "\n")








