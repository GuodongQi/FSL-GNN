import numpy as np
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def shuffle(image_set, label, init_label, fetch_global=False):
    index = np.random.permutation(len(image_set))
    image_set = image_set[index]
    if not fetch_global:
        label = deepcopy(init_label)
    label = label[index]
    return image_set, label


def show_images(image_batch, label_batch):
    cols = 5
    rows = image_batch.shape[0] // cols + 1
    gs = gridspec.GridSpec(cols, rows)
    fig = plt.figure(figsize=(84 * rows, 84 * cols), dpi=2)
    for j in range(image_batch.shape[0]):
        plt.subplot(gs[j])
        plt.axis('off')
        img = image_batch[j].type(torch.uint8).permute(1,2,0).cpu().numpy()
        plt.imshow(img)
        # plt.title("%s"%label_batch[j])
    plt.show()
    print(label_batch)
