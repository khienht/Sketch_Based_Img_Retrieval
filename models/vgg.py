import torch.nn as nn
import torch as t
import torchvision.models as models


class MyVGG16(nn.Module):
    def __init__(self, pretrained=True):
        super(MyVGG16, self).__init__()

        vgg16 = models.vgg16(pretrained=pretrained)
        self.features = vgg16.features
        self.avgpool = vgg16.avgpool

        # Lấy danh sách các lớp con của classifier
        classifier_children = list(vgg16.classifier.children())

        # Thay đổi lớp thứ 0 thành một lớp mới
        classifier_children[0] = nn.Linear(in_features=512*7*7, out_features=4096)

        # Tạo lại classifier với các lớp đã chỉnh sửa
        self.classifier = nn.Sequential(
            *classifier_children[:-1],
            nn.Linear(4096, 1000)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = t.flatten(x, 1)
        x = self.classifier(x)

        return x