from collections import defaultdict

import csv
import cv2
import lmdb
import numpy as np
import os
import pyarrow
from tqdm import tqdm
from nvidia.dali.pipeline import Pipeline
from nvidia.dali import ops, types

from configs.config import configs


class DALI_MiniImagenet:
    def __init__(self, path_root, t_task, n_way, k_shot, k_query, split, fetch_global=False):
        """
        MiniImagenet dataset with Nvidia-DALI to accelerate data processing
        :param path_root: dataset path
        :param t_task: num of tasks
        :param n_way:  num of classes
        :param k_shot: num of examples each class in support set
        :param k_query:  num of examples each class in query set
        :param split:  if 'train' then train dataset. "val", "test"
        :param fetch_global: whether fetch global or local class label index
        """
        self.t_task = t_task
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.split = split
        self.path_root = path_root
        self.fet_global = fetch_global

        self.path = os.path.join(path_root, 'images')

        self.lmdb_file = os.path.join(path_root, "lmdb_data", "%s.lmdb" % self.split)
        if not os.path.exists(self.lmdb_file):
            print("lmdb_file is not found, start to generate %s" % self.lmdb_file)
            # generate lmdb dataset
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
                (self.support_set_size + self.query_set_size) * self.t_task)  # how many episode

    def __iter__(self):
        return self

    def __next__(self):
        """
        create a num of tasks  data
        """
        env = self.env

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
                    support_x.append(np.frombuffer(res[0], np.uint8))
                    support_y.append(np.array([self.label2num[res[1]]]))

                for i, img_id in enumerate(query_imgs):
                    res = pyarrow.deserialize(txn.get(u'{}'.format(img_id).encode('ascii')))
                    query_x.append(np.frombuffer(res[0], np.uint8))
                    query_y.append(np.array([self.label2num[res[1]]]))
            support_x = np.array(support_x)
            query_x = np.array(query_x)
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

        total_query_x = np.hstack(total_query_x)
        total_query_y = np.hstack(total_query_y)
        total_support_x = np.hstack(total_support_x)
        total_support_y = np.hstack(total_support_y)

        return np.hstack([total_support_x, total_query_x]).tolist(), \
               np.hstack([total_support_y, total_query_y]).reshape([-1, 1])

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


class MiniPipeline(Pipeline):
    def __init__(self, data_iter, batch_size, x_dim, num_threads, device_id):
        super(MiniPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        data_iter = iter(data_iter)
        self.source = ops.ExternalSource(source=data_iter, num_outputs=2)
        self.decode = ops.ImageDecoder(device='mixed', output_type=types.RGB)
        self.rrc = ops.RandomResizedCrop(device='gpu', size=x_dim[:2], random_area=[0.95, 1.0])
        self.flip = ops.Flip(device='gpu')
        # self.colortwist = ops.ColorTwist(device='gpu', brightness=.1, contrast=.1, saturation=.1, hue=.1)
        self.norm = ops.CropMirrorNormalize(device='gpu',
                                            output_layout=types.NCHW,
                                            # mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            # std=[0.229 * 255, 0.224 * 255, 0.225 * 255]
                                            )

    def define_graph(self):
        images, label = self.source()
        images = self.decode(images)
        images = self.rrc(images)
        # images = self.flip(images)
        # images = self.colortwist(images)
        images = self.norm(images)

        return images, label


if __name__ == '__main__':
    cfg = configs
    train_data = DALI_MiniImagenet(cfg.data_path, cfg.t_task, cfg.n_way, cfg.k_shot, cfg.k_query, 'train')

    # train_data.__next__()
    pipe = MiniPipeline(train_data, cfg.t_task * cfg.n_way * (cfg.k_shot + cfg.k_query), cfg.x_dim, num_threads=2,
                        device_id=7)
    pipe.build()
    pipe_out = pipe.run()
    for i in range(1):
        print(pipe_out[0].as_cpu().at(i).shape, pipe_out[1].at(i))
