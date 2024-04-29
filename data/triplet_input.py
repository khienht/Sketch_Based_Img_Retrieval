import os
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random


def find_classes(root):
    classes = [d for d in os.listdir(root)]
    classes.sort()
    class_to_idex = {classes[i]: i for i in range(len(classes))}
    index_to_class = {i: classes[i] for i in range(len(classes))}
    return classes, class_to_idex, index_to_class

def make_dataset(root):
    images = []

    cnames = os.listdir(root)
    for cname in cnames:
        c_path = os.path.join(root, cname)
        fnames = os.listdir(c_path)
        for fname in fnames:
            path = os.path.join(c_path, fname)
            images.append(path)

    return images

def make_dataset1(root, cname):
    images = []
    c_path = os.path.join(root, cname)
    fnames = os.listdir(c_path)
    for fname in fnames:
        path = os.path.join(c_path, fname)
        images.append(path)

    return images

class TripleDataset(data.Dataset):
    def __init__(self, photo_root, sketch_root):
        super(TripleDataset, self).__init__()
        self.tranform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        classes, class_to_idx, idx_to_class = find_classes(photo_root)

        self.photo_root = photo_root
        self.sketch_root = sketch_root

        self.photo_paths = sorted(make_dataset(self.photo_root))
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class

        self.len = len(self.photo_paths)

    def __getitem__(self, index):

        photo_path = self.photo_paths[index]
        sketch_path, label = self._getrelate_sketch(photo_path)

        if label == 0:
            r = list(range(label+1,125))
        else:
            r = list(range(0,label))+list(range(label+1,125))
        i = random.choice(r) #class

        neg_path, label1 = self._getneg_photo(i)
        
        photo = Image.open(photo_path).convert('RGB')
        sketch = Image.open(sketch_path).convert('RGB')
        neg = Image.open(neg_path).convert('RGB')
        
        P = self.tranform(photo)
        A = self.tranform(sketch)
        L = label
        N = self.tranform(neg)
        L1 = label1

        return {'A': A, 'P': P, 'N': N, 'L': L, 'L1': L1}

    def __len__(self):
        return self.len
    
    def __iter__(self):
        # Trả về một trình lặp để lặp qua từng phần tử của dataset
        return iter(self[i] for i in range(len(self)))
    
    def _getrelate_sketch(self, photo_path):
        paths = photo_path.split('\\')
        fname = paths[-1].split('.')[0]
        cname = paths[-2]

        label = self.class_to_idx[cname]

        sketchs = sorted(os.listdir(os.path.join(self.sketch_root, cname)))

        sketch_rel = []
        for sketch_name in sketchs:
            if sketch_name.split('-')[0] == fname:
                sketch_rel.append(sketch_name)

        rnd = np.random.randint(0, len(sketch_rel))

        sketch = sketch_rel[rnd]

        return os.path.join(self.sketch_root, cname, sketch), label
    
    def _getneg_photo(self, i):
        cname = self.idx_to_class[i]
        neg_path = os.path.join(self.photo_root, cname)
        paths = make_dataset1(self.photo_root, cname)

        ran = random.randint(0, 79)

        neg_img = paths[ran]
        label = self.class_to_idx[cname]

        return neg_img, cname
