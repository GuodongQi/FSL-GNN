import os
import pickle as pkl
from collections import defaultdict

import cv2
import lmdb
import numpy as np
import platform
import pyarrow
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm


def f1(x):  # multi-progress for windows
    img_decode = np.fromstring(x, np.uint8)
    img_data = cv2.imdecode(img_decode, cv2.IMREAD_COLOR)
    return Image.fromarray(img_data[:, :, ::-1])


class TierImageNet(Dataset):
    """
        put tierimagenet files as:
            test_images_png.pkl
            test_labels.pkl
            train_images_png.pkl
            train_labels.pkl
            val_images_png.pkl
            val_labels.pkl

            # better way to preserve numpy arrays on disk is lmdb. So we convert image.pkl to .lmdb
            test_images.npz (converted)
            train_images.npz (converted)
            val_images.npz (converted)
    """

    def __init__(self, path_root, n_way, k_shot, k_query, x_dim, split, augment='0', test=None, shuffle=True):
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.x_dim = list(map(int, x_dim.split(',')))
        self.split = split
        self.shuffle = shuffle
        self.path_root = path_root

        if augment == '0':
            self.transform = transforms.Compose([
                transforms.Lambda(f1),
                transforms.Resize(self.x_dim[:2]),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        elif augment == '1':
            if self.split == 'train':
                self.transform = transforms.Compose([
                    # lambda x: Image.open(x).convert('RGB'),
                    transforms.Lambda(f1),
                    transforms.Resize((self.x_dim[0] + 20, self.x_dim[1] + 20)),
                    transforms.RandomCrop(self.x_dim[:2]),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.1),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
            else:
                self.transform = transforms.Compose([
                    # lambda x: Image.open(x).convert('RGB'),
                    transforms.Lambda(f1),
                    transforms.Resize((self.x_dim[0] + 20, self.x_dim[1] + 20)),
                    transforms.RandomCrop(self.x_dim[:2]),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])

        self.lmdb_file = os.path.join(path_root, "lmdb_data", "%s.lmdb" % self.split)
        if not os.path.exists(self.lmdb_file):
            print("lmdb_file is not found, start to generate %s" % self.lmdb_file)
            self._generate_lmdb()

        # read lmdb_file
        self.env = lmdb.open(self.lmdb_file, subdir=False,
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.total_sample = pyarrow.deserialize(txn.get(b'__len__'))
            self.keys = pyarrow.deserialize(txn.get(b'__keys__'))
            self.total_cls = pyarrow.deserialize(txn.get(b'__total_cls__'))

        self.image_labels = [i.decode() for i in self.keys]
        self.dic_img_label = defaultdict(list)
        for i in self.image_labels:
            self.dic_img_label[int(i.split('_')[0])].append(i)

        self.support_set_size = self.n_way * self.k_shot  # num of samples per support set
        self.query_set_size = self.n_way * self.k_query

        self.episode = self.total_sample // (self.support_set_size + self.query_set_size)  # how many episode

        # create episodes
        self.episode_sets = []

        if platform.system().lower() == 'windows':
            self.platform = "win"
            del self.env
        elif platform.system().lower() == 'linux':
            self.platform = "linux"

    def __getitem__(self, episode):
        """Given one episode,  read support set and query set """
        if self.platform == "win":
            env = lmdb.open(self.lmdb_file, subdir=False,
                            readonly=True, lock=False,
                            readahead=False, meminit=False)
        else:
            env = self.env

        support_x = torch.FloatTensor(self.support_set_size, self.x_dim[2], self.x_dim[0], self.x_dim[1])
        query_x = torch.FloatTensor(self.query_set_size, self.x_dim[2], self.x_dim[0], self.x_dim[1])
        support_y = []
        query_y = []

        support_imgs = []
        query_imgs = []
        # select n_way classes randomly
        selected_classes = np.random.choice(self.total_cls, self.n_way)
        # select k_shot + k_query for each class
        for selected_class in selected_classes:
            selected_imgs = np.random.choice(
                self.dic_img_label[self.num2label[selected_class]], self.k_shot + self.k_query, False)
            support_imgs += selected_imgs[:self.k_shot].tolist()
            query_imgs += selected_imgs[self.k_shot:].tolist()

        with env.begin(write=False) as txn:
            for i, img_id in enumerate(support_imgs):
                res = pyarrow.deserialize(txn.get(u'{}'.format(img_id).encode('ascii')))
                support_x[i] = self.transform(res[0])
                support_y.append(res[1])

            for i, img_id in enumerate(query_imgs):
                res = pyarrow.deserialize(txn.get(u'{}'.format(img_id).encode('ascii')))
                query_x[i] = self.transform(res[0])
                query_y.append(res[1])

        support_y = np.array(support_y)
        query_y = np.array(query_y)

        if self.shuffle:
            index = np.random.permutation(len(support_y))
            support_x = support_x[index]
            if not self.fet_global:
                support_y = np.array([i for i in range(self.n_way) for j in range(self.k_shot)])
            support_y = support_y[index]

            index = np.random.permutation(len(query_y))
            query_x = query_x[index]
            if not self.fet_global:
                query_y = np.array([i for i in range(self.n_way) for j in range(self.k_query)])
            query_y = query_y[index]

        return \
            support_x, torch.LongTensor(torch.Tensor(support_y).long()), \
            query_x, torch.LongTensor(torch.Tensor(query_y).long())

    def __len__(self):
        return self.episode

    def _dumps_pyarrow(self, obj):
        """
        Serialize an object.
        Returns:
            Implementation-dependent bytes-like object
        """
        return pyarrow.serialize(obj).to_buffer()

    def _generate_lmdb(self, write_frequency=40000):
        # load csv data, which consists of image_name and image_label, and convert to lmdb file
        lmdb_dir = os.path.join(self.path_root, "lmdb_data")
        if not os.path.exists(lmdb_dir):
            os.mkdir(lmdb_dir)
        if self.split == "train":
            map_size = int(9e9)
        else:
            map_size = int(4.2e9)
        db = lmdb.open(self.lmdb_file, subdir=False, map_size=map_size, readonly=False, meminit=False, map_async=True)
        txn = db.begin(write=True)

        image_pkl = os.path.join(self.path_root, '%s_images_png.pkl' % self.split)
        cache_path_labels = os.path.join(self.path_root, '%s_labels.pkl' % self.split)
        assert os.path.exists(image_pkl), '%s is not existed, please check it' % image_pkl
        # load images_png.pkl
        with open(image_pkl, 'rb') as f:
            array = pkl.load(f, encoding='latin1')
        # load .npz image and .pkl label
        print("\tload cached labels from {}".format(cache_path_labels))
        with open(cache_path_labels, 'rb') as f:
            data = pkl.load(f)
            label_specific = data["label_specific"]
            # label_general = data["label_general"]
            # label_specific_str = data["label_specific_str"]
            # label_general_str = data["label_general_str"]
        keys = []
        for ii, item in enumerate(tqdm(array, desc='converting database  %s' % self.lmdb_file)):
            # im = cv2.imdecode(item, cv2.IMREAD_COLOR)
            # im = np.array(im[:, :, ::-1])  # BGR to RGB
            im = item.tostring()
            image_name = '%d_%09d' % (label_specific[ii], ii)
            keys.append(u'{}'.format(image_name).encode('ascii'))
            txn.put(key=u'{}'.format(image_name).encode('ascii'),
                    value=self._dumps_pyarrow((im, label_specific[ii])))
            if ii % write_frequency == 0:
                txn.commit()
                txn = db.begin(write=True)

        # finish iterating through dataset
        txn.commit()
        with db.begin(write=True) as txn:
            txn.put(b'__keys__', self._dumps_pyarrow(keys))
            txn.put(b'__len__', self._dumps_pyarrow(len(keys)))
            txn.put(b'__total_cls__', self._dumps_pyarrow(label_specific.max() + 1))
        print("Flushing database ...")
        db.sync()
        db.close()

# if __name__ == '__main__':
#     a = TierImageNet("F:\\dataset\\tiered-imagenet", 2, 2, 1, 20, 'test')
#     print(a[0])
