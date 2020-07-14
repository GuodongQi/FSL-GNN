import os
import time

from math import exp
import numpy as np
import platform
import torch
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs.config import my_configs as configs
from core.BackBone import ConvNet, ResNet12, ResNet18, WRN
from core.GNN import GNN, Memory
from datasets.data_loader import data_loader
from collections import defaultdict

# train on the GPU or on the CPU, if a GPU is not available
from datasets.data_utils import shuffle, show_images

configs.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#
# if torch.cuda.is_available() and platform.system().lower() == 'linux':
#     configs.use_dali = True
#  torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(4)


def build_net():
    if configs.backbone == "ConvNet":
        backbone = ConvNet(configs.emb_size)
    elif configs.backbone == "ResNet12":
        backbone = ResNet12(configs.emb_size)
    elif configs.backbone == "ResNet18":
        backbone = ResNet18(configs.emb_size)
    elif configs.backbone == "WRN":
        backbone = WRN(configs.emb_size)
    else:
        raise NotImplementedError

    gnn = GNN(configs)
    mem = Memory(configs)
    gnn.to(configs.device)
    backbone.to(configs.device)
    mem.to(configs.device)
    return backbone, gnn, mem


def decompose_to_input(train_data):
    if configs.use_dali:
        images, labels = train_data[0]['data'], train_data[0]['label']
    else:
        images, labels = train_data[0][0], train_data[1][0]

    support_x = images[:configs.t_task * configs.n_way * configs.k_shot]
    query_x = images[configs.t_task * configs.n_way * configs.k_shot:]
    support_y = labels[:configs.t_task * configs.n_way * configs.k_shot]
    query_y = labels[configs.t_task * configs.n_way * configs.k_shot:]
    # convert label to one-hot
    support_y = torch.zeros(configs.t_task * configs.n_way * configs.k_shot, configs.n_way) \
        .scatter(1, support_y.view(-1, 1), 1)
    query_y = torch.zeros(configs.t_task * configs.n_way * configs.k_query, configs.n_way) \
        .scatter(1, query_y.view(-1, 1), 1)

    inputs = [support_x.to(configs.device),
              support_y.to(configs.device),
              query_x.to(configs.device),
              query_y.to(configs.device)]
    return inputs


def decompose_embedding_from_backbone(embedding, labels, embedding_global=None):
    embedding = torch.unsqueeze(embedding, 1)
    support_embedding = embedding[:configs.t_task * configs.n_way * configs.k_shot].reshape(
        [configs.t_task, configs.n_way * configs.k_shot, configs.emb_size])
    query_embedding = embedding[configs.t_task * configs.n_way * configs.k_shot:].reshape(
        [configs.t_task, configs.n_way * configs.k_query, configs.emb_size])
    support_y = labels[0].reshape([configs.t_task * configs.n_way * configs.k_shot, configs.n_way])
    query_y = labels[1].reshape([configs.t_task * configs.n_way * configs.k_query, configs.n_way])
    if embedding_global is not None:
        embedding_global = torch.unsqueeze(embedding_global, 1)
        support_embedding_glo = embedding_global[:configs.t_task * configs.n_way * configs.k_shot].reshape(
            [configs.t_task, configs.n_way * configs.k_shot, configs.emb_size])
        query_embedding_glo = embedding_global[configs.t_task * configs.n_way * configs.k_shot:].reshape(
            [configs.t_task, configs.n_way * configs.k_query, configs.emb_size])
        return support_embedding, query_embedding, support_y, query_y, support_embedding_glo, query_embedding_glo

    else:
        return support_embedding, query_embedding, support_y, query_y


def main():
    alpha = configs.v_loss_rate
    beta = configs.k_loss_rate
    gama = configs.cls_loss_rate
    # for exp
    for_exp = True

    pwd = os.getcwd()
    save_path = os.path.join(pwd, configs.save_path,
                             "%d_mem_size_%d_way_%d_shot_%d_query" % (
                                 configs.mem_size, configs.n_way, configs.k_shot, configs.k_query))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_path = os.path.join(save_path, "model")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # init dataloader
    print("init data loader")
    train_db, total_train = data_loader(configs, num_workers=configs.num_workers, split="train",
                                        use_dali=configs.use_dali)
    val_db, total_valid = data_loader(configs, num_workers=configs.num_workers, split="val",
                                      use_dali=configs.use_dali)

    # init neural networks
    backbone, gnn, mem = build_net()
    params = list(backbone.parameters()) + list(gnn.parameters()) + list(mem.parameters())

    print(repr(configs))
    # print(backbone, gnn, mem)

    # optimizer
    if configs.train_optim == 'adam':
        optimizer = optim.Adam(params, lr=configs.lr, weight_decay=configs.weight_decay)
    elif configs.train_optim == 'sgd':
        optimizer = optim.SGD(
            params, lr=configs.lr, weight_decay=configs.weight_decay, momentum=configs.momentum)
    elif configs.train_optim == 'rmsprop':
        optimizer = optim.RMSprop(
            params, lr=configs.lr, weight_decay=configs.weight_decay, momentum=configs.momentum,
            alpha=0.9, centered=True)
    else:
        raise Exception("error optimizer")

    # learning rate decay policy
    if configs.lr_policy == 'multi_step':
        scheduler = MultiStepLR(optimizer, milestones=list(map(int, configs.milestones.split(','))),
                                gamma=configs.lr_gama)
    elif configs.lr_policy == 'exp':
        scheduler = ExponentialLR(optimizer, gamma=configs.lr_gamma)
    elif configs.lr_policy == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=configs.lr_gama, patience=80, verbose=True)
    else:
        raise Exception('error lr decay policy')

    if configs.start_epoch:
        check_point = torch.load(os.path.join(model_path, "%d_model.pkl" % configs.start_epoch))
        backbone.load_state_dict(check_point['backbone_state_dict'])
        gnn.load_state_dict(check_point['gnn_state_dict'])
        mem.load_state_dict(check_point['mem_state_dict'])
        if for_exp:
            scheduler.load_state_dict(check_point['scheduler'])
            optimizer.load_state_dict(check_point['optimizer'])
        print('Loading Parameters from %d_model.pkl' % configs.start_epoch)

    # Train and validation
    best_acc = 0.0
    best_loss = np.inf
    wait = 0

    writer = SummaryWriter(os.path.join(save_path, "logs", "%s" % time.strftime('%Y-%m-%d-%H-%M')))

    for ep in range(configs.start_epoch, configs.epochs):
        margin = 2

        thresh_train = [1 * (1 / (1 + exp(-ep / 100 + margin))), 1 - (1 / (1 + exp(-ep / 100 + margin))),
                        1 * (1 / (1 + exp(-ep / 100 + margin))), 1 - (1 / (1 + exp(-ep / 100 + margin))), ]
        print("epoch:", ep, "thresh_train:", thresh_train)
        loss_print = defaultdict(list)

        train_loss_item = 0
        train_acc_item = 0
        train_loss_k = 0
        train_loss_v = 0
        train_loss_c = 0

        train_pbar = tqdm(train_db, total=total_train)
        for step, train_data in enumerate(train_pbar):
            train_pbar.set_description(
                'train_epoc:{}, total_loss:{:.5f}, acc:{:.5f}, loss_k:{:.5f}, loss_v:{:.5f}, loss_c:{:.5f}'.format(
                    ep, train_loss_item, train_acc_item, train_loss_k, train_loss_v, train_loss_c))

            # start to train
            backbone.train()
            gnn.train()
            mem.train()
            support_x, support_y, query_x, query_y = decompose_to_input(train_data)
            # propagation
            embedding, global_embedding = backbone(torch.cat([support_x, query_x], 0) / 255.0)
            support_embedding, query_embedding, support_y, query_y, sup_emb_glo, que_emb_glo = \
                decompose_embedding_from_backbone(embedding, [support_y, query_y], global_embedding)
            embedding, global_embedding, loss_k, loss_v, loss_s = mem([support_embedding, query_embedding],
                                                              [sup_emb_glo, que_emb_glo], thresh_train)
            loss_cls, acc = gnn(embedding, global_embedding, [support_y, query_y])
            loss = alpha * loss_v + beta * loss_k + gama * loss_cls + loss_s
            # for visual images and labels
            '''
            imgs = torch.cat([support_x, query_x], 0)
            labels = torch.cat([support_y, query_y], 0)
            labels = torch.argmax(labels, -1)
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            from collections import defaultdict
            rows = configs.n_way
            batch_size = imgs.size(0)
            cols = batch_size // rows
            gs = gridspec.GridSpec(rows, cols)
            fig = plt.figure(figsize=(84 * cols, 84 * rows), dpi=2)
            plt.rc('font', size=8)
            nums = defaultdict(int)
            for j in range(batch_size):
                label = int(labels[j] + 0)
                plt.subplot(gs[label*cols+nums[label]])
                nums[label] += 1
                plt.axis('off')
                img = imgs[j].type(torch.uint8).permute(1, 2, 0).cpu().numpy()
                plt.imshow(img)
            print(repr(labels))
            plt.savefig('test.jpg')
            '''

            train_loss_item = loss.item()
            train_acc_item = acc.item()
            train_loss_c = loss_cls.item() * gama
            train_loss_k = loss_k.item() * alpha
            train_loss_v = loss_v.item() * beta
            loss_print["train_loss"].append(train_loss_item)
            loss_print["train_loss_c"].append(train_loss_c)
            loss_print["train_loss_k"].append(train_loss_k)
            loss_print["train_loss_v"].append(train_loss_v)
            loss_print["train_acc"].append(train_acc_item)
            loss_print["simil"].append(train_acc_item)

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm(model.parameters(), 4.0)
            optimizer.step()

        # for valid
        valid_loss_item = 0
        valid_acc_item = 0
        valid_loss_k = 0
        valid_loss_v = 0
        valid_loss_c = 0
        valid_pbar = tqdm(val_db, total=total_valid)
        for step, train_data in enumerate(valid_pbar):
            valid_pbar.set_description(
                'valid_epoc:{}, total_loss:{:.5f}, acc:{:.5f}, loss_k:{:.5f}, loss_v:{:.5f}, loss_c:{:.5f}'.format(
                    ep, valid_loss_item, valid_acc_item, valid_loss_k, valid_loss_v, valid_loss_c))
            # start to valid
            backbone.eval()
            gnn.eval()
            mem.eval()
            support_x, support_y, query_x, query_y = decompose_to_input(train_data)
            # propagation
            with torch.no_grad():
                embedding, global_embedding = backbone(torch.cat([support_x, query_x], 0) / 255.0)
                support_embedding, query_embedding, support_y, query_y, sup_emb_glo, que_emb_glo = \
                    decompose_embedding_from_backbone(embedding, [support_y, query_y], global_embedding)
                embedding, global_embedding, loss_k, loss_v, loss_s = mem([support_embedding, query_embedding],
                                                                  [sup_emb_glo, que_emb_glo], thresh_train)
                loss_cls, acc = gnn(embedding, global_embedding, [support_y, query_y])
                loss = alpha * loss_v + beta * loss_k + gama * loss_cls + loss_s

            valid_loss_item = loss.item()
            valid_acc_item = acc.item()
            valid_loss_c = loss_cls.item() * gama
            valid_loss_k = loss_k.item() * alpha
            valid_loss_v = loss_v.item() * beta
            loss_print["valid_loss"].append(train_loss_item)
            loss_print["valid_loss_c"].append(valid_loss_c)
            loss_print["valid_loss_k"].append(valid_loss_k)
            loss_print["valid_loss_v"].append(valid_loss_v)
            loss_print["valid_acc"].append(valid_acc_item)

        scheduler.step(np.mean(loss_print["valid_loss"]))
        print('epoch:{}, lr:{:.6f}'.format(ep, optimizer.param_groups[0]['lr']))
        print(["{}:{:.6f}".format(key, np.mean(loss_print[key])) for key in loss_print.keys()])

        # tensorboard
        # writer.add_graph(net, (inputs,))
        for key in loss_print.keys():
            writer.add_scalar(key, np.mean(loss_print[key]), ep)
        writer.add_scalar('Loss/train', np.mean(loss_print["train_loss"]), ep)
        writer.add_scalar('Loss/val', np.mean(loss_print["valid_loss"]), ep)
        writer.add_scalar('Accuracy/train', np.mean(loss_print["train_acc"]), ep)
        writer.add_scalar('Accuracy/val', np.mean(loss_print["valid_acc"]), ep)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], ep)

        # Model Save and Stop Criterion
        cond1 = (np.mean( loss_print["valid_acc"]) > best_acc)
        cond2 = (np.mean( loss_print["valid_loss"]) < best_loss)

        if cond1 or cond2:
            best_acc = np.mean(loss_print["valid_acc"])
            best_loss = np.mean(loss_print["valid_loss"])
            print('best val loss:{:.5f}, acc:{:.5f}'.format(best_loss, best_acc))

            # save model
            torch.save(save_state(for_exp, backbone, gnn, mem, optimizer, scheduler),
                       os.path.join(save_path, "model", '%d_model.pkl' % ep))
            wait = 0

        else:
            wait += 1
            if (ep + 1) % 100 == 0:
                torch.save(save_state(for_exp, backbone, gnn, mem, optimizer, scheduler),
                           os.path.join(save_path, "model", '%d_model.pkl' % ep))

        if wait > configs.patience:
            break


def save_state(for_exp, backbone, gnn, mem, optimizer=None, scheduler=None):
    if for_exp:
        save_dict = {
            'backbone_state_dict': backbone.state_dict(),
            'gnn_state_dict': gnn.state_dict(),
            'mem_state_dict': mem.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
    else:
        save_dict = {
            'backbone_state_dict': backbone.state_dict(),
            'gnn_state_dict': gnn.state_dict(),
            'mem_state_dict': mem.state_dict(),
        }
    return save_dict


if __name__ == '__main__':
    main()
