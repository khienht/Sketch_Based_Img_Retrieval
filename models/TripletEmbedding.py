import torch as t
from torch import nn
from data import TripleDataLoader
from models.vgg import MyVGG16
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
import gc
import torchvision.models as models
import torchvision.transforms as transforms



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
        model = MyVGG16(pretrained=pretrained)
        # model.classifier[0] = nn.Linear(in_features=512*7*7, out_features=4096, bias=True)
        # model.classifier[2] = nn.Dropout()
        # model.classifier[5] = nn.Dropout()
        # model.classifier[6] = nn.Linear(in_features=4096, out_features=125, bias=True)
        # for name, module in model.named_modules():
        #     module_type = type(module)
        #     if module_type == t.nn.functional:
        #         print(f"{name} is a Functional layer")

        return model

    def _get_resnet50(self, pretrained=True):
        model = resnet50(pretrained=pretrained)
        model.fc = nn.Linear(in_features=2048, out_features=125)

        return model

    def train(self):
        if self.net == 'vgg16':
            photo_net = DataParallel(self._get_vgg16())
            sketch_net = DataParallel(self._get_vgg16())
        elif self.net == 'resnet50':
            photo_net = DataParallel(self._get_resnet50())
            sketch_net = DataParallel(self._get_resnet50())
        # photo_net.to(device)
        # sketch_net.to(device)

        # if self.fine_tune:
        #     photo_net_root = self.model_root
        #     sketch_net_root = self.model_root.replace('photo', 'sketch')

        #     photo_net.load_state_dict(t.load(photo_net_root, map_location=t.device('cpu')))
        #     sketch_net.load_state_dict(t.load(sketch_net_root, map_location=t.device('cpu')))

        print('net')
        print(photo_net)

        # triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=self.p)
        photo_cat_loss = nn.CrossEntropyLoss()
        sketch_cat_loss = nn.CrossEntropyLoss()

        my_triplet_loss = TripletLoss()

        # optimizer
        photo_optimizer = t.optim.Adam(photo_net.parameters(), lr=self.lr)
        sketch_optimizer = t.optim.Adam(sketch_net.parameters(), lr=self.lr)

        # if self.vis:
        #     vis = Visualizer(self.env)

        triplet_loss_meter = AverageValueMeter()
        sketch_cat_loss_meter = AverageValueMeter()
        photo_cat_loss_meter = AverageValueMeter()

        data_loader = TripleDataLoader(self.dataloader_opt)
        dataset = data_loader.load_data()
        print(dataset.__len__())

        for epoch in range(self.epochs):

            print('---------------{0}---------------'.format(epoch))

            if self.test and epoch % self.test_f == 0:

                # tester_config = Config()
                # tester_config.test_bs = 128
                # tester_config.photo_net = photo_net
                # tester_config.sketch_net = sketch_net

                # tester_config.photo_test = self.photo_test
                # tester_config.sketch_test = self.sketch_test

                # tester = Tester(tester_config)
                # test_result = tester.test_instance_recall()

                # result_key = list(test_result.keys())
                # vis.plot('recall', np.array([test_result[result_key[0]], test_result[result_key[1]]]),
                #               legend=[result_key[0], result_key[1]])
                if self.save_model:
                    t.save(photo_net.state_dict(), self.save_dir + '/photo' + '/photo_' + self.net + '_%s.pth' % epoch)
                    t.save(sketch_net.state_dict(), self.save_dir + '/sketch' + '/sketch_' + self.net + '_%s.pth' % epoch)

            photo_net.train()
            sketch_net.train()

            for i in range(len(dataset)):
                data = dataset[i]
                photo_optimizer.zero_grad()
                sketch_optimizer.zero_grad()

                photo = data['P']
                # print(photo.shape)

                photo = photo.unsqueeze(0)
                sketch = data['S']
                sketch = sketch.unsqueeze(0)
                label = data['L']
                label = t.unsqueeze(t.tensor(label), 0)
                
                p_cat, p_feature = photo_net(photo)
                s_cat, s_feature = sketch_net(sketch)

                # category loss
                p_cat_loss = photo_cat_loss(p_cat, label)
                s_cat_loss = sketch_cat_loss(s_cat, label)

                photo_cat_loss_meter.add(p_cat_loss.item())
                sketch_cat_loss_meter.add(s_cat_loss.item())

                # triplet loss
                loss = p_cat_loss + s_cat_loss

                # tri_record = 0.
                '''
                for i in range(self.batch_size):
                    # negative
                    negative_feature = t.cat([p_feature[0:i, :], p_feature[i + 1:, :]], dim=0)
                    # print('negative_feature.size :', negative_feature.size())
                    # photo_feature
                    anchor_feature = s_feature[i, :]
                    anchor_feature = anchor_feature.expand_as(negative_feature)
                    # print('anchor_feature.size :', anchor_feature.size())

                    # positive
                    positive_feature = p_feature[i, :]
                    positive_feature = positive_feature.expand_as(negative_feature)
                    # print('positive_feature.size :', positive_feature.size())

                    tri_loss = triplet_loss(anchor_feature, positive_feature, negative_feature)

                    tri_record = tri_record + tri_loss

                    # print('tri_loss :', tri_loss)
                    loss = loss + tri_loss
                '''
                # print('tri_record : ', tri_record)

                my_tri_loss = my_triplet_loss(s_feature, p_feature) / (self.batch_size - 1)
                triplet_loss_meter.add(my_tri_loss.item())
                # print('my_tri_loss : ', my_tri_loss)

                # print(tri_record - my_tri_loss)
                loss = loss + my_tri_loss
                # print('loss :', loss)
                # loss = loss / opt.batch_size

                loss.backward()

                # update param for model
                photo_optimizer.step()
                sketch_optimizer.step()

                # if self.vis:
                #     vis.plot('triplet_loss', np.array([triplet_loss_meter.value()[0], photo_cat_loss_meter.value()[0],
                #                                        sketch_cat_loss_meter.value()[0]]),
                #              legend=['triplet_loss', 'photo_cat_loss', 'sketch_cat_loss'])

                triplet_loss_meter.reset()
                photo_cat_loss_meter.reset()
                sketch_cat_loss_meter.reset()









