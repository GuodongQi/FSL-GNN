import torch.nn as nn
import torch

from core.BackBone import ConvNet
from core.utils import compute_similarity, compute_loss


class CalWeights(nn.Module):
    def __init__(self, in_channels, hidden=128) -> None:
        """
        Given nodes of GNN, compute adjacent matrix
        :param in_channels: in_channel number
        :param hidden: hidden  number
        """
        super().__init__()
        self.hidden = hidden
        self.in_channels = in_channels

        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                              out_channels=self.hidden,
                                              kernel_size=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=self.hidden,
                                              out_channels=int(self.hidden * 1.5),
                                              kernel_size=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=int(self.hidden * 1.5)),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=int(self.hidden * 1.5),
                                              out_channels=int(self.hidden * 1.5),
                                              kernel_size=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=int(self.hidden * 1.5)),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=int(self.hidden * 1.5),
                                              out_channels=self.hidden,
                                              kernel_size=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_5 = nn.Sequential(nn.Conv2d(in_channels=self.hidden,
                                              out_channels=1,
                                              kernel_size=1,
                                              bias=True),
                                    nn.BatchNorm2d(num_features=1),
                                    )

    def forward(self, nodes, device, activation="softmax"):
        """
        :param nodes: GNN node, [t_task,n_way*k_shot+n_way*k_query, emb_size]
        :param device: GPU CPU
        :param activation: activation function
        :return: weight matrix [t_task, 1, n_way*k_shot+n_way*k_query]
        """
        nodes1 = nodes.reshape([nodes.size(0), nodes.size(1), 1, nodes.size(2)])
        nodes2 = nodes.reshape([nodes.size(0), 1, nodes.size(1), nodes.size(2)])
        nodes = torch.abs(nodes1 - nodes2)  # [t_task,n_way*k_shot+n_way*k_query, n_way*k_shot+n_way*k_query,emb_size]
        nodes = nodes.permute(0, 3, 1, 2)  # NCHW
        # use CNN as Metric
        # [t_task, 1, n_way*k_shot+n_way*k_query, n_way*k_shot+n_way*k_query]
        w = self.conv_1(nodes)
        w = self.conv_2(w)
        w = self.conv_3(w)
        w = self.conv_4(w)
        w = self.conv_5(w)

        # for softmax
        w_id = torch.eye(w.size(-1)).unsqueeze(0).unsqueeze(0).expand_as(w).to(device)
        if activation == "softmax":
            w = w - 1e8 * w_id  # to ensure identity=0
            w = nn.functional.softmax(w, -1)
        elif activation == "sigmoid":
            size = w.size()
            w = w.reshape(-1)
            w = torch.sigmoid(w)
            w = w.reshape(size)
            w *= (1 - w_id)
        elif activation == 'none':
            w *= (1 - w_id)
        else:
            raise NotImplementedError

        w = torch.cat([w, w_id], 1)  # introduce the nodes themselves feature

        return w  # [t_task, 2, n_way*k_shot+n_way*k_query, n_way*k_shot+n_way*k_query]


class UpdateNode(nn.Module):

    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        # set size
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.layer_last = nn.Sequential(nn.Linear(in_features=2 * self.in_channel,
                                                  out_features=self.out_channel, bias=True),
                                        nn.BatchNorm1d(self.out_channel))

    def forward(self, nodes, weights):
        """
        update node with node@weights
        :param nodes:  [t_task,n_way*k_shot+n_way*k_query, in_channel]
        :param weights: [t_task, 2, n_way*k_shot+n_way*k_query, n_way*k_shot+n_way*k_query]
        :return: new nodes
        """
        new_nodes = nodes.unsqueeze(1).expand(-1, 2, -1, -1)  # [t_task, 2, n_way*k_shot+n_way*k_query, in_channel]
        new_nodes = torch.matmul(weights, new_nodes)  # [t_task, 2, n_way*k_shot+n_way*k_query, in_channel]
        # [t_task, n_way*k_shot+n_way*k_query, 2*in_channel]
        new_nodes = torch.cat([new_nodes[:, 0, :, :], new_nodes[:, 1, :, :]], -1)
        # [t_task*(n_way*k_shot+n_way*k_query), 2*in_channel]
        new_nodes = torch.reshape(new_nodes, [-1, new_nodes.size(-1)])
        # [t_task*(n_way*k_shot+n_way*k_query), out_channel]
        new_nodes = self.layer_last(new_nodes)
        # [t_task, n_way*k_shot+n_way*k_query, out_channel]
        new_nodes = torch.reshape(new_nodes, [nodes.size(0), nodes.size(1), new_nodes.size(-1)])
        # shortcut, using cat [t_task, n_way*k_shot+n_way*k_query, in_channel+out_channel]
        new_nodes = torch.cat([new_nodes, nodes], -1)
        return new_nodes


class Memory(nn.Module):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_way = self.cfg.n_way
        self.emb_size = self.cfg.emb_size
        self.device = self.cfg.device
        self.second_size = self.cfg.emb_size
        # memory setting
        self.mem_size = self.cfg.mem_size
        self.thresh = self.cfg.thresh
        self.q_v = self.cfg.q_v
        self.q_k = self.cfg.q_k
        self.margin = self.cfg.margin

        # memory blocks/slots
        self.memory_keys = nn.Parameter(
            pow(1 / self.emb_size, 0.5) * torch.ones(self.mem_size, self.emb_size, dtype=torch.float).to(self.device),
            requires_grad=True)
        self.memory_values = nn.Parameter(
            pow(1 / self.emb_size, 0.5) * torch.ones(self.mem_size, self.second_size, dtype=torch.float).to(self.device),
            requires_grad=True)

    def forward(self, embedding, embedding_global, metric="Cosine"):
        """
        Given embedding input, we update keys and value with embedding and global_embedding
        used to weight sum to last layer embedding
        :param embedding: [t_task, n_way*k_shot, edm_size] +  [t_task, n_way*k_query, edm_size]
        :param embedding_global: [t_task, n_way*k_shot, edm_size] +  [t_task, n_way*k_query, edm_size]
        :param metric: Euclidean distance or cosine distance
        :return: a robust memory and weighted sum embedding
        """
        norm_emb = nn.functional.normalize(torch.cat(embedding, 1), p=2, dim=-1)
        norm_emb_glo = nn.functional.normalize(torch.cat(embedding_global, 1), p=2, dim=-1)

        # for key->value, generate reasonable global embedding
        # 1. calculate distance and score between memory_key and embedding
        similarity_kv = compute_similarity(norm_emb, self.memory_keys, metric=metric)
        # 2. We update global value of memory, where similarity_kv greater than threshold.
        score = torch.where(torch.ge(similarity_kv, self.thresh), similarity_kv,
                            torch.zeros_like(similarity_kv).to(self.device))  # [t_task, n_way*k_shot]
        memory_values_updated = nn.functional.normalize(
            torch.mean(self.q_k * self.memory_values + (1 - self.q_k) * score.unsqueeze(-1) * norm_emb_glo.unsqueeze(-2),
                       [0, 1]), p=2, dim=-1)

        # for value->key
        # 3. calculate distance and score between memory_key and embedding
        similarity_vk = compute_similarity(norm_emb_glo, self.memory_values, metric=metric)
        # 4. find max similarity_vk, update the corresponding keys with support embedding
        score = torch.zeros(similarity_vk.shape).to(self.device).scatter(-1, torch.argmax(similarity_vk, -1).
                                                                         unsqueeze(-1), 1.0)
        memory_keys_updated = nn.functional.normalize(
            torch.mean(self.q_k * self.memory_keys + (1 - self.q_k) * score.unsqueeze(-1) * norm_emb.unsqueeze(-2),
                       [0, 1]), p=2, dim=-1)

        # # apply update memory
        # self.memory_keys.data = memory_keys_updated
        # self.memory_values.data = memory_values_updated

        # 5. based on updated memory, select global features for each example
        similarity_kv = compute_similarity(norm_emb, memory_keys_updated, metric=metric)
        similarity_vk = compute_similarity(norm_emb_glo, memory_values_updated, metric=metric)

        embedding_global = norm_emb_glo + torch.sum(
            similarity_kv.unsqueeze(-1) * memory_values_updated.unsqueeze(0).unsqueeze(0), -2)
        embedding_global = nn.functional.normalize(embedding_global, p=2, dim=-1)

        # Calculate loss with self-supervised
        # 1. key->value
        score_max = torch.zeros(similarity_kv.shape).to(self.device). \
            scatter(-1, torch.argmax(similarity_kv, -1).unsqueeze(-1), 1.0)
        select_global = torch.sum(score_max.unsqueeze(-1) * memory_values_updated.unsqueeze(0).unsqueeze(0), 2)
        loss_v = compute_loss(select_global, norm_emb_glo)

        # 2. value->key
        score_max = torch.zeros(similarity_vk.shape).to(self.device).scatter(-1,
                                                                             torch.argmax(similarity_vk, -1).unsqueeze(
                                                                                 -1), 1.0)
        score_min = torch.zeros(similarity_vk.shape).to(self.device).scatter(-1,
                                                                             torch.argmin(similarity_vk, -1).unsqueeze(
                                                                                 -1), 1.0)
        select_max = torch.sum(score_max.unsqueeze(-1) * memory_keys_updated.unsqueeze(0).unsqueeze(0), 2)
        select_min = torch.sum(score_min.unsqueeze(-1) * memory_keys_updated.unsqueeze(0).unsqueeze(0), 2)
        loss_k = torch.max(compute_loss(select_max, norm_emb) - compute_loss(select_min, norm_emb) + self.margin,
                           torch.tensor(0, dtype=torch.float).to(self.device))
        # 3. prototype loss
        # todo: Use prototype
        return torch.cat([norm_emb, embedding_global], -1), loss_k, loss_v


class GNN(nn.Module):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_channel = 2 * cfg.emb_size + cfg.n_way
        self.out_chanel = cfg.out_channel
        self.num_layers = cfg.num_layers
        self.cfg_device = cfg.device
        self.device = self.cfg_device
        self.hidden = cfg.hidden

        for i in range(self.num_layers):
            # create Weights.
            # input size: current node channel, [t_task,n_way*k_shot+n_way*k_query, in_channel]
            # output size: [t_task, 2, n_way*k_shot+n_way*k_query, n_way*k_shot+n_way*k_query]
            self.add_module('w_{}'.format(i), CalWeights(self.in_channel + i * self.out_chanel, self.hidden))
            # create GNN layer.
            # input size: [t_task,n_way*k_shot+n_way*k_query, in_channel]
            # output size: [t_task,n_way*k_shot+n_way*k_query, in_channel+out_channel]
            self.add_module('l_{}'.format(i),
                            UpdateNode(self.in_channel + i * self.out_chanel, out_channel=self.out_chanel))

        self.fc = nn.Sequential(
            nn.Linear(self.in_channel + self.num_layers * self.out_chanel, self.cfg.n_way, bias=True),
            # nn.Linear(self.out_chanel, self.cfg.n_way, bias=True),
            # nn.Softmax(-1)
        )

    def forward(self, embedding, labels):
        """
        Feed embedding to GNN
        :param embedding: embedding feature for images
        :param labels: images label, one-hot
        :return: prediction
        """
        support_embedding = embedding[:, :self.cfg.n_way * self.cfg.k_shot, :]
        query_embedding = embedding[:, self.cfg.n_way * self.cfg.k_shot:, :]
        support_y = labels[0]
        query_y = labels[1]

        support_x = torch.cat([support_embedding, support_y.reshape(
            [self.cfg.t_task, self.cfg.n_way * self.cfg.k_shot, self.cfg.n_way])], -1)
        query_x = torch.cat([query_embedding,
                             torch.zeros([self.cfg.t_task, self.cfg.n_way * self.cfg.k_query, self.cfg.n_way]).to(
                                 self.cfg.device)], -1)
        x = torch.cat([support_x, query_x], 1)

        for i in range(self.num_layers):
            w = self._modules['w_{}'.format(i)](x, self.cfg.device)
            x = self._modules['l_{}'.format(i)](x, w)
        # pred = x[..., -self.out_chanel:]
        pred = x
        pred = self.fc(pred)
        pred_support = pred[:, :self.cfg.n_way * self.cfg.k_shot, :].reshape([-1, self.cfg.n_way])
        pred_query = pred[:, self.cfg.n_way * self.cfg.k_shot:, :].reshape([-1, self.cfg.n_way])
        ce = nn.CrossEntropyLoss().to(self.cfg.device)
        # loss
        only_use_train = False
        if only_use_train:
            gts = torch.argmax(support_y, -1)
            loss = ce(pred_support, gts)
        else:
            gts = torch.argmax(torch.cat([support_y, query_y], 0), -1)
            loss = ce(torch.cat([pred_support, pred_query], 0), gts)

        # acc
        query_node = torch.argmax(pred_query, -1)
        query_gt = torch.argmax(query_y, -1)

        correct = (query_node == query_gt).sum()
        total = self.cfg.t_task * self.cfg.n_way * self.cfg.k_query
        acc = 1.0 * correct.float() / float(total)
        return loss, acc


if __name__ == '__main__':
    class A:
        def __init__(self) -> None:
            self.device = 'cuda:0'
            self.emb_size = 64
            self.out_channel = 64
            self.hidden = 128
            self.num_layers = 3


    cfg = A()
    emb = torch.rand([3, 5 * (5 + 5), 64]).to(cfg.device)
    g = GNN(cfg).to(cfg.device)
    print(g)
    out = g(emb)
