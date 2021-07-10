import os
import time
import logging

import numpy as np
from tqdm import tqdm

# for clustering and 2d representations
from sklearn import random_projection

import torch
from torch import optim
from torch.optim import lr_scheduler
import torchvision

# refer to https://www.guruguru.science/competitions/17/discussions/a39d588e-aff2-4728-8323-b07f15563552/
# contrastive leaning dataset, loss support
import lightly

from engine.trainer import Trainer

from models.simple_siam_net import SiamNet

from losses.negative_cosine_similality import CosineContrastiveLoss

from utils.helper import Save_Handle, AverageMeter, worker_init_fn
from utils.visualizer import GraphPlotter, get_scatter_plot_with_thumbnails

class SimSiamTrainer(Trainer):
    def setup(self):
        """initialize the datasets, model, loss and optimizer"""
        args = self.args
        self.graph = GraphPlotter(self.save_dir, ['NegCosineSim'], 'simsiam')

        self.data_dir = args.data_dir
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        # define the augmentations for self-supervised learning
        collate_fn = lightly.data.ImageCollateFunction(
            input_size=args.crop_size,
            # require invariance to flips and rotations
            hf_prob=0.5,
            vf_prob=0.5,
            rr_prob=0.5,
            # satellite images are all taken from the same height
            # so we use only slight random cropping
            min_scale=0.5,
            # use a weak color jitter for invariance w.r.t small color changes
            cj_prob=0.2,
            cj_bright=0.1,
            cj_contrast=0.1,
            cj_hue=0.1,
            cj_sat=0.1,
        )

        # create a lightly dataset for training, since the augmentations are handled
        # by the collate function, there is no need to apply additional ones here
        dataset_train_simsiam = lightly.data.LightlyDataset(
            input_dir=self.data_dir
        )

        # create a dataloader for training
        self.train_dataloader = torch.utils.data.DataLoader(
            dataset_train_simsiam,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=args.num_workers
        )

        # create a torchvision transformation for embedding the dataset after training
        # here, we resize the images to match the input size during training and apply
        # a normalization of the color channel based on statistics from imagenet
        val_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((args.crop_size, args.crop_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=lightly.data.collate.imagenet_normalize['mean'],
                std=lightly.data.collate.imagenet_normalize['std'],
            )
        ])

        # create a lightly dataset for embedding
        val_dataset = lightly.data.LightlyDataset(
            input_dir=self.data_dir,
            transform=val_transforms
        )

        # create a dataloader for embedding
        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers
        )

        # for the scatter plot we want to transform the images to a two-dimensional
        # vector space using a random Gaussian projection
        self.embedding_projection = random_projection.GaussianRandomProjection(n_components=2)

        # Define model, loss, optim
        self.model = SiamNet(arch=args.arch)
        print(self.model)
        self.model.to(self.device)

        self.criterion = CosineContrastiveLoss()
        self.criterion.to(self.device)

        lr = 0.05 * args.batch_size / 256

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4
        )

        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.max_epoch)

        self.start_epoch = 0
        self.best_loss = np.inf
        
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))

        self.save_list = Save_Handle(max_num=args.max_model_num)

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            self.epoch = epoch
            self.train_epoch(epoch)
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch(epoch)

    def train_epoch(self, epoch):
        epoch_loss = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode

        for (input1, input2), _, _ in tqdm(self.train_dataloader, ncols=60):
            input1 = input1.to(self.device)
            input2 = input2.to(self.device)

            with torch.set_grad_enabled(True):
                (z1, z2), (p1, p2) = self.model(input1, input2)
                loss = self.criterion(z1, z2, p1, p2)
                epoch_loss.update(loss.item(), input1.size(0))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

        logging.info('Epoch {} Train, Loss: {:.5f}, lr: {:.5f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), self.optimizer.param_groups[0]['lr'], time.time()-epoch_start))
        
        self.graph(self.epoch, [epoch_loss.get_avg()])

        if epoch % self.args.check_point == 0:
            model_state_dic = self.model.state_dict()
            save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
            torch.save({
                'epoch': self.epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_state_dict': model_state_dic
            }, save_path)
            self.save_list.append(save_path)  # control the number of saved models

    def val_epoch(self, epoch):
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_loss = AverageMeter()

        embeddings = []
        filenames = []
        for input1, _, fnames in tqdm(self.val_dataloader, ncols=60):
            input1 = input1.to(self.device)

            with torch.set_grad_enabled(False):
                z = self.model(input1, None, test=True)
                z = z.squeeze()

            embeddings.append(z)
            filenames = filenames + list(fnames)

        embeddings = torch.cat(embeddings, dim=0)
        embeddings = embeddings.cpu().numpy()
        embeddings_2d = self.embedding_projection.fit_transform(embeddings)
        # normalize the embeddings to fit in the [0, 1] square
        M = np.max(embeddings_2d, axis=0)
        m = np.min(embeddings_2d, axis=0)
        embeddings_2d = (embeddings_2d - m) / (M - m)

        logging.info('Epoch {} Val, Loss: {:.5f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), time.time()-epoch_start))

        if epoch % 10 == 0:
            get_scatter_plot_with_thumbnails(epoch, embeddings_2d, self.data_dir, self.save_dir, filenames)

        model_state_dic = self.model.state_dict()
        if self.best_loss > epoch_loss.get_avg():
            self.best_loss = epoch_loss.get_avg()
            logging.info("save min loss {:.2f} model epoch {}".format(self.best_loss, self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))