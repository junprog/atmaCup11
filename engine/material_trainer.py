import os
import time
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms

from engine.trainer import Trainer

from datasets.atma_dataset import atmaDataset
from datasets.one_hot_encode import one_hot_encode

from models.resnet import ResNet

from utils.helper import Save_Handle, AverageMeter
from utils.visualizer import GraphPlotter
from utils.metrics import calc_accuracy

class MaterialTrainer(Trainer):
    def setup(self):
        """initialize the datasets, model, loss and optimizer"""
        args = self.args
        self.graph = GraphPlotter(self.save_dir, ['BCEwithlogits', 'accuracy'], 'material_classify')

        self.data_dir = args.data_dir
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        train_csv_path = os.path.join(self.data_dir, 'train.csv')
        #test_csv_path = os.path.join(self.data_dir, 'test.csv')
        material_path = os.path.join(self.data_dir, 'material.csv')
        #techniques_path = os.path.join(self.data_dir, 'techniques.csv')
        self.img_path = os.path.join(self.data_dir, 'photos')

        self.kf = KFold(n_splits=5)

        # Define transform
        self.train_df = pd.read_csv(train_csv_path)
        self.material_df = pd.read_csv(material_path)
        self.encoded_material_df = one_hot_encode(self.material_df)

        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Define model, loss, optim
        self.model = ResNet(arch=args.arch, num_classes=25)
        print(self.model)
        self.model.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.criterion.to(self.device)

        lr = 0.1 * args.batch_size / 256

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

        for train, val in self.kf(self.encoded_material_df):
            train_dataset = atmaDataset(
                data_dir = self.img_path,
                img_name_df = self.encoded_material_df.object_id[train],
                target_df = self.encoded_material_df.drop('object_id', axis=1)[train],
                trans = self.train_transforms
            )

            val_dataset = atmaDataset(
                data_dir = self.img_path,
                img_name_df = self.encoded_material_df.object_id[val],
                target_df = self.encoded_material_df.drop('object_id', axis=1)[val],
                trans = self.val_transforms
            )

            self.train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True, drop_last=True, num_workers=0)
            self.val_loader = torch.utils.data.DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=0)

            for epoch in range(self.start_epoch, args.max_epoch):
                logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
                self.epoch = epoch
                self.train_epoch(epoch)
                if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                    self.val_epoch(epoch)

    def train_epoch(self, epoch):
        epoch_loss = AverageMeter()
        epoch_acc = AverageMeter()

        epoch_start = time.time()
        self.model.train()  # Set model to training mode

        for inputs, target in tqdm(self.train_dataloader, ncols=60):
            inputs = inputs.to(self.device)
            target = target.to(self.device)

            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, target)

                epoch_loss.update(loss.item(), inputs.size(0))
                epoch_acc.update(calc_accuracy(outputs.item(), target.item()), inputs.size(0))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

        logging.info('Epoch {} Train, Acc: {:.5f}, Loss: {:.5f}, lr: {:.5f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_acc.get_avg(), epoch_loss.get_avg(), self.optimizer.param_groups[0]['lr'], time.time()-epoch_start))
        
        self.graph(self.epoch, [epoch_loss.get_avg(), epoch_acc.get_avg()])

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
        epoch_acc = AverageMeter()

        for inputs, target in tqdm(self.val_dataloader, ncols=60):
            inputs = inputs.to(self.device)

            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, target)

            epoch_loss.update(loss.item(), inputs.size(0))
            epoch_acc.update(calc_accuracy(outputs.item(), target.item()), inputs.size(0))

        logging.info('Epoch {} Val, Acc: {:.5f}, Loss: {:.5f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_acc.get_avg(), epoch_loss.get_avg(), time.time()-epoch_start))

        model_state_dic = self.model.state_dict()
        if self.best_loss > epoch_loss.get_avg():
            self.best_loss = epoch_loss.get_avg()
            logging.info("save min loss {:.2f} model epoch {}".format(self.best_loss, self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))