import csv
import os
import platform
from collections import defaultdict

import cv2
import lmdb
import numpy as np
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


class MiniImagenet(Dataset):
    """
        put mini-imagenet files as:
             :
            |- images/*.jpg includes all images
            |- converted_lmdb/lmdb file
            |- train.csv
            |- test.csv
            |- val.csv

        The LMDB file storage format:

            key | value
            --- | ---
            img-id1 | (jpeg_raw1, label1)
            img-id2 | (jpeg_raw2, label2)
            img-id3 | (jpeg_raw3, label3)
            ... | ...
            img-idn | (jpeg_rawn, labeln)
            `__keys__` | [img-id1, img-id2, ... img-idn]
            `__len__` | n
            '__label2num__' | n
            '__num2label__' | n

    """

    def __init__(self, path_root, t_task, n_way, k_shot, k_query, x_dim, split, augment='0', test=None, shuffle=True,
                 fetch_global=False):
        self.t_task = t_task
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.x_dim = list(map(int, x_dim.split(',')))
        self.split = split
        self.shuffle = shuffle
        self.path_root = path_root
        self.fet_global = fetch_global

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

        self.path = os.path.join(path_root, 'images')

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
            self.label2num = pyarrow.deserialize(txn.get(b'__label2num__'))
            self.num2label = pyarrow.deserialize(txn.get(b'__num2label__'))

        self.image_labels = [i.decode() for i in self.keys]
        self.total_cls = len(self.num2label)
        self.dic_img_label = defaultdict(list)
        for i in self.image_labels:
            self.dic_img_label[i[:9]].append(i)

        self.support_set_size = self.n_way * self.k_shot  # num of samples per support set
        self.query_set_size = self.n_way * self.k_query

        self.episode = self.total_sample // (
                    self.t_task * (self.support_set_size + self.query_set_size))  # how many episode

        if platform.system().lower() == 'windows':
            self.platform = "win"
            del self.env
        elif platform.system().lower() == 'linux':
            self.platform = "linux"

    def __getitem__(self, cur_episode):
        """Given one episode, read support set and query set of this episode  """
        if self.platform == "win":
            env = lmdb.open(self.lmdb_file, subdir=False,
                            readonly=True, lock=False,
                            readahead=False, meminit=False)
        else:
            env = self.env
        # episode_set = self.episode_sets[episode]
        total_support_x = []
        total_query_x = []
        total_support_y = []
        total_query_y = []

        for t in range(self.t_task):
            # create a task (n_way*k_shot+ n_way*k_query)

            support_x = []
            query_x = []
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
                    support_x.append(self.transform(res[0]))
                    support_y.append(np.array([self.label2num[res[1]]]))

                for i, img_id in enumerate(query_imgs):
                    res = pyarrow.deserialize(txn.get(u'{}'.format(img_id).encode('ascii')))
                    query_x.append(self.transform(res[0]))
                    query_y.append(np.array([self.label2num[res[1]]]))
            support_x = torch.stack(support_x, 0)
            query_x = torch.stack(query_x, 0)
            support_y = np.array(support_y)
            query_y = np.array(query_y)

            # shuffle:
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

            # a batch
            total_query_x.append(query_x)
            total_query_y.append(query_y)
            total_support_x.append(support_x)
            total_support_y.append(support_y)

        total_query_x = torch.cat(total_query_x, 0)
        total_query_y = np.hstack(total_query_y)
        total_support_x = torch.cat(total_support_x, 0)
        total_support_y = np.hstack(total_support_y)

        imgs = torch.cat([total_support_x, total_query_x], 0)
        labels = torch.from_numpy(np.hstack([total_support_y, total_query_y]).reshape([-1, 1]))
        return imgs, labels

    def __len__(self):
        return self.episode

    def _dumps_pyarrow(self, obj):
        """
        Serialize an object.
        Returns:
            Implementation-dependent bytes-like object
        """
        return pyarrow.serialize(obj).to_buffer()

    def _generate_lmdb(self, write_frequency=5000):
        # load csv data, which consists of image_name and image_label, and convert to lmdb file
        lmdb_dir = os.path.join(self.path_root, "lmdb_data")
        if not os.path.exists(lmdb_dir):
            os.mkdir(lmdb_dir)
        if self.split == "train":
            map_size = int(5e9)
        else:
            map_size = int(1e9)
        db = lmdb.open(self.lmdb_file, subdir=False, map_size=map_size, readonly=False, meminit=False, map_async=True)
        txn = db.begin(write=True)
        label2num = {}
        num2label = {}
        # load csv data
        with open(os.path.join(self.path_root, self.split + '.csv')) as csv_file:
            csvreader = csv.reader(csv_file, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            j = 0
            keys = []
            for i, row in enumerate(tqdm(csvreader, desc="generating lmdb")):
                image_name, image_label = row
                keys.append(u'{}'.format(image_name).encode('ascii'))
                if image_label not in label2num.keys():
                    label2num[image_label] = j
                    num2label[j] = image_label
                    j += 1
                img = cv2.imread(os.path.join(self.path, image_name))
                img_encode = cv2.imencode('.jpg', img)[1]
                img_encode = img_encode.tostring()
                txn.put(key=u'{}'.format(image_name).encode('ascii'),
                        value=self._dumps_pyarrow(
                            # (np.asarray(Image.open(os.path.join(self.path, image_name)).convert('RGB')), image_label))
                            (img_encode, image_label)
                        ))
                if i % write_frequency == 0:
                    txn.commit()
                    txn = db.begin(write=True)

            # finish iterating through dataset
            txn.commit()
            with db.begin(write=True) as txn:
                txn.put(b'__keys__', self._dumps_pyarrow(keys))
                txn.put(b'__len__', self._dumps_pyarrow(len(keys)))
                txn.put(b'__num2label__', self._dumps_pyarrow(num2label))
                txn.put(b'__label2num__', self._dumps_pyarrow(label2num))
            print("Flushing database ...")
            db.sync()
            db.close()
