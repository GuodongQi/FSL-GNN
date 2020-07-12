import argparse

configs = argparse.ArgumentParser()


def str2bool(s):
    s = s.lower()
    true_set = {'yes', 'true', 't', 'y', '1'}
    false_set = {'false', 'no', 'f', 'n', '0'}
    if s in true_set:
        return True
    elif s in false_set:
        return False
    else:
        raise ValueError('Excepted {}'.format(' '.join(true_set | false_set)))


# dataset
configs.add_argument('--data_path', type=str, default="/media/data1/datasets/FSL_datasets/MiniImageNet",
                     help="dataset path root")
configs.add_argument('--dataset', type=str, default='miniImagenet', help="miniImagenet or tierImagenet")
# configs.add_argument('--data_path', type=str, default="F:\\dataset\\tiered-imagenet", help="dataset path root")
# configs.add_argument('--dataset', type=str, default='tierImagenet', help="miniImagenet or tierImagenet")
configs.add_argument('--num_workers', type=int, default=8, help="dataloader num_works")

# GPU

# FSL setting
configs.add_argument('--t_task', type=int, default=5, help="T-task")
configs.add_argument('--n_way', type=int, default=5, help="N-way")
configs.add_argument('--k_shot', type=int, default=5, help="K-shot")
configs.add_argument('--k_query', type=int, default=1, help="K-query")
configs.add_argument('--fetch_global', type=str2bool, default=False, help="fetch global label or one hot label")

configs.add_argument('--epochs', type=int, default=1000, help="epoch")

# BackBone
configs.add_argument('--backbone', type=str, default="ConvNet", help="ConvNet, ResNet12, ResNet18, WRN")

# memory setting
configs.add_argument('--mem_size', type=int, default=15, help='memory size')
configs.add_argument('--thresh', type=float, default=0.3, help='select which score')
configs.add_argument('--q_k', type=float, default=0.99, help='memory key update momentum')
configs.add_argument('--q_v', type=float, default=0.999, help='memory value update momentum')
configs.add_argument('--margin', type=float, default=1, help='cosine similarity margin')

# loss weights
configs.add_argument('--v_loss_rate', type=float, default=0.1, help='loss value ratio')
configs.add_argument('--k_loss_rate', type=float, default=0.1, help='loss key ration')
configs.add_argument('--cls_loss_rate', type=float, default=1, help='loss class ratio')


# network setting
configs.add_argument('--x_dim', type=str, default="84,84,3", metavar='XDIM', help='input image dims')
configs.add_argument('--emb_size', type=int, default=64, metavar='embedding size', help='embedding size')
configs.add_argument('--hidden', type=int, default=64, help="dimensionality of hidden layers (default: 64)")
configs.add_argument('--out_channel', type=int, default=128,  help="dimensionality of output channels (default: 128)")
configs.add_argument('--num_layers', type=int, default=3,  help="Number of GNN layers")



# optimization params
configs.add_argument('--train_optim', type=str, default="adam",
                     help="optimizer: adam,sgd,rmsprop")
configs.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                     help="base learning rate")
configs.add_argument('--weight_decay', type=float, default=0.0005,
                     help="weight_decay")
configs.add_argument('--momentum', type=float, default=0.9,
                     help="momentum")

# learning rate decay policy
configs.add_argument('--lr_policy', type=str, default="plateau",
                     help="lr decay policy: multi_step, exponentialLR, Plateau")
configs.add_argument('--milestones', type=str, default="100,200,300",
                     help="milestone learning rate decay")
configs.add_argument('--step_size', type=int, default=1000, metavar='STEPSIZE',
                     help="lr decay step size")
configs.add_argument('--lr_gama', type=float, default=0.5, metavar='GAMMA',
                     help="decay rate")
configs.add_argument('--patience', type=int, default=200, metavar='PATIENCE',
                     help="train patience until stop")


# save and restore params

configs.add_argument('--start_epoch', type=int, default=0, metavar='start_epoch',
                     help="epoch to restore params")
configs.add_argument('--save_path', type=str, default="checkpoints")

# gpu DALI
configs.add_argument('--use_dali', type=str2bool, default=True)

# shuffle data
configs.add_argument('--shuffle', type=str2bool, default=True)

my_configs = configs.parse_args()
