import torch


def compute_loss(x1, x2):
    return 1 - torch.sum(x1 * x2, -1)


def find_idx(similarity, thresh, device):
    """return minimal positive and maximal negative similarity score index"""
    thresh_pos, thresh_neg = thresh
    pos = torch.where(torch.ge(similarity, thresh_pos), similarity, 2 * torch.ones_like(similarity).to(device))
    min_pos_idx = torch.argmin(pos, -1)
    neg = torch.where(torch.lt(similarity, thresh_neg), similarity, torch.zeros_like(similarity).to(device))
    max_neg_idx = torch.argmax(neg, -1)

    # some query vector may not has positive memory slot, we dont calculate its loss
    pos_score = torch.gather(pos, dim=-1, index=min_pos_idx.unsqueeze(-1)).squeeze(-1)
    pos_score = torch.where(pos_score != 2, torch.ones_like(pos_score).to(device),
                            torch.zeros_like(pos_score).to(device))

    neg_score = torch.gather(neg, dim=-1, index=max_neg_idx.unsqueeze(-1)).squeeze(-1)
    neg_score = torch.where(neg_score > 0, torch.ones_like(neg_score).to(device),
                            torch.zeros_like(neg_score).to(device))

    return min_pos_idx, max_neg_idx, pos, pos_score, neg_score


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
