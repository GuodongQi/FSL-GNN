import torch


def compute_loss(x1, x2):
    return 1 - torch.sum(x1 * x2, -1).mean()


def compute_similarity(x1, x2, metric="Euclidean"):
    # broadcast
    # x1: t_task, n_way*(k_shot+k_query), embedding
    # x2: n_way, embedding
    x1 = x1.unsqueeze(2)
    x2 = x2.unsqueeze(0).unsqueeze(0)
    if metric == "Euclidean":
        # Euclidean distance
        distance = torch.norm(x1 - x2, -1)
        similarity = torch.reciprocal(distance + 1e-8)
    elif metric == "Cosine":
        similarity = torch.sum(x1 * x2, -1)
    else:
        raise NotImplementedError
    return similarity
