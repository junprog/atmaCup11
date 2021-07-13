import os

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import offsetbox as osb
from matplotlib import rcParams as rcp

# for resizing images to thumbnails
import torchvision.transforms.functional as functional

## metrics epoch のグラフ表示
class GraphPlotter:
    def __init__(self, save_dir, metrics: list, suffix):
        self.save_dir = save_dir
        self.graph_name = 'result_{}.png'.format(suffix)
        self.metrics = metrics

        self.epochs = []
        
        self.value_dict = dict()
        for metric in metrics:
            self.value_dict[metric] = []

    def __call__(self, epoch, values: list):
        assert (len(values) == len(self.value_dict)), 'metrics and values length shoud be same size.'
    
        self.epochs.append(epoch)

        fig, ax = plt.subplots()
    
        for i, metric in enumerate(self.metrics):
            self.value_dict[metric].append(values[i])
            ax.plot(self.epochs, self.value_dict[metric], label=metric)
        
        plt.ylim(-0.5,2.5)
        ax.legend(loc=0)
        fig.tight_layout()  # レイアウトの設定
        fig.savefig(os.path.join(self.save_dir, self.graph_name))

        plt.title(self.metrics)
        plt.close()

## Embedding 表示
def get_scatter_plot_with_thumbnails(epoch, embeddings_2d, path_to_data, save_dir, filenames):
    """Creates a scatter plot with image overlays.
    """
    # initialize empty figure and add subplot
    fig = plt.figure(figsize=(12,12))
    fig.suptitle('SimSiam Scatter Plot')
    ax = fig.add_subplot(1, 1, 1)
    # shuffle images and find out which images to show
    shown_images_idx = []
    shown_images = np.array([[1., 1.]])
    iterator = [i for i in range(embeddings_2d.shape[0])]
    np.random.shuffle(iterator)
    for i in iterator:
        # only show image if it is sufficiently far away from the others
        dist = np.sum((embeddings_2d[i] - shown_images) ** 2, 1)
        if np.min(dist) < 1.5e-3:
            continue
        shown_images = np.r_[shown_images, [embeddings_2d[i]]]
        shown_images_idx.append(i)

    # plot image overlays
    for idx in shown_images_idx:
        thumbnail_size = int(rcp['figure.figsize'][0] * 5.)
        path = os.path.join(path_to_data, filenames[idx])
        img = Image.open(path)
        img = functional.resize(img, thumbnail_size)
        img = np.array(img)
        img_box = osb.AnnotationBbox(
            osb.OffsetImage(img, cmap=plt.cm.gray_r),
            embeddings_2d[idx],
            pad=0.2,
        )
        ax.add_artist(img_box)

    # set aspect ratio
    ratio = 1. / ax.get_data_ratio()
    ax.set_aspect(ratio, adjustable='box')

    fig.savefig(os.path.join(save_dir, '{}_embed.png'.format(epoch)))
    plt.close()