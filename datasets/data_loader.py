import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import my_configs as configs
from datasets.miniImagenet import MiniImagenet
from datasets.tierImagesnet import TierImageNet
from datasets.dali_utils.miniImagenet import DALI_MiniImagenet, MiniPipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator


def data_loader(cfg, num_workers=4, split='train', use_dali=False):
    if not use_dali:
        if cfg.dataset == "miniImagenet":
            train_data = MiniImagenet(cfg.data_path, cfg.t_task, cfg.n_way, cfg.k_shot, cfg.k_query, cfg.x_dim, split)
        elif cfg.dataset == 'tierImagenet':
            train_data = TierImageNet(cfg.data_path, cfg.t_task, cfg.n_way, cfg.k_shot, cfg.k_query, cfg.x_dim, split)
        else:
            raise Exception("check your spelling of dataset")
        train_db = DataLoader(train_data, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

    else:
        if cfg.dataset == "miniImagenet":
            train_data = DALI_MiniImagenet(cfg.data_path, cfg.t_task, cfg.n_way, cfg.k_shot, cfg.k_query, split)
        elif cfg.dataset == 'tierImagenet':
            train_data = TierImageNet(cfg.data_path, cfg.t_task, cfg.n_way, cfg.k_shot, cfg.k_query, split)
        else:
            raise Exception("check your spelling of dataset")

        pipe = MiniPipeline(train_data, cfg.t_task * cfg.n_way * (cfg.k_shot + cfg.k_query), cfg.x_dim,
                            num_threads=num_workers,
                            device_id=0)
        train_db = DALIClassificationIterator(pipe, size=train_data.total_sample, fill_last_batch=False,
                                              last_batch_padded=True, auto_reset=True)
    return train_db, train_data.episode


if __name__ == '__main__':
    print(configs.__repr__())
    db, total = data_loader(configs, 32, use_dali=True)
    for i in range(100):
        for content in db:
            print(content)

            # print(j,i)
            # support_y = i[1]
            # support_y = torch.zeros(configs.n_way * configs.k_shot, configs.n_way).scatter(1, support_y.view(-1, 1), 1)
            pass
