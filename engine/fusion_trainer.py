import os
import time
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import transforms

from engine.trainer import Trainer
from datasets.atma_dataset import AtmaDataset
from models.fusion_net import FcFusionNet_v1, FcFusionNet_v2 
from utils.helper import Save_Handle, AverageMeter
from utils.visualizer import GraphPlotter

class FusionTrainer(Trainer):
    def setup(self):
        """initialize the datasets, model, loss and optimizer"""
        args = self.args

        self.data_dir = args.data_dir
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        train_csv_path = os.path.join(self.data_dir, 'train.csv')
        #test_csv_path = os.path.join(self.data_dir, 'test.csv')
        #material_path = os.path.join(self.data_dir, 'materials.csv')
        #techniques_path = os.path.join(self.data_dir, 'techniques.csv')
        self.img_path = os.path.join(self.data_dir, 'photos')

        self.skf = StratifiedKFold(n_splits=5)

        # Define transform
        self.train_df = pd.read_csv(train_csv_path)

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

        # Define loss
        self.criterion = nn.MSELoss()
        self.criterion.to(self.device)

        self.save_list = Save_Handle(max_num=args.max_model_num)

    def train(self):
        """training process"""
        args = self.args

        for i, (train, val) in enumerate(self.skf.split(self.train_df['object_id'], self.train_df['target'])):

            if not os.path.exists(os.path.join(self.save_dir, 'cv_' + str(i))):
                os.mkdir(os.path.join(self.save_dir, 'cv_' + str(i)))

            self.tr_graph = GraphPlotter(os.path.join(self.save_dir, 'cv_' + str(i)), ['MSE'], 'train')
            self.vl_graph = GraphPlotter(os.path.join(self.save_dir, 'cv_' + str(i)), ['MSE'], 'val')

            train_dataset = AtmaDataset(
                data_dir = self.img_path,
                img_name_df = self.train_df.object_id[train],
                target_df = self.train_df.drop('object_id', axis=1).drop('sorting_date', axis=1).drop('art_series_id', axis=1).loc[train],
                trans = self.train_transforms
            )

            val_dataset = AtmaDataset(
                data_dir = self.img_path,
                img_name_df = self.train_df.object_id[val],
                target_df = self.train_df.drop('object_id', axis=1).drop('sorting_date', axis=1).drop('art_series_id', axis=1).loc[val],
                trans = self.val_transforms
            )

            self.train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
            self.val_loader = torch.utils.data.DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

            # Define model, scheduler, optim
            mate_weight_path = os.path.join(args.mate_res_dir, 'cv_' + str(i), 'best_model.pth')
            tech_weight_path = os.path.join(args.tech_res_dir, 'cv_' + str(i), 'best_model.pth')

            self.model = FcFusionNet_v2(
                arch=args.arch,
                mate_out_num=6,
                tech_out_num=3,
                simsiam_weight_path=args.init_weight_path,
                mate_weight_path=mate_weight_path,
                tech_weight_path=tech_weight_path,
                freeze=True
            )
            print(self.model)
            self.model.to(self.device)

            lr = 0.01

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

            #self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[25, 50, 75, 90], gamma=0.1)
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.max_epoch)

            self.start_epoch = 0
            self.best_loss = np.Inf
            
            if args.resume:
                suf = args.resume.rsplit('.', 1)[-1]
                if suf == 'tar':
                    checkpoint = torch.load(args.resume, self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.start_epoch = checkpoint['epoch'] + 1
                elif suf == 'pth':
                    self.model.load_state_dict(torch.load(args.resume, self.device))

            for epoch in range(self.start_epoch, args.max_epoch):
                logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
                self.epoch = epoch

                self.train_epoch(epoch, i)
                self.scheduler.step()

                if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                    self.val_epoch(epoch, i)

    def train_epoch(self, epoch, i):
        epoch_loss = AverageMeter()

        epoch_start = time.time()
        self.model.train()  # Set model to training mode

        ## unfreeze all params at 51 epoch
        if epoch == 51:
           self.model.unfreeze()

        for inputs, target in tqdm(self.train_loader, ncols=60):
            inputs = inputs.to(self.device)
            target = target.to(self.device)

            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, target)

                epoch_loss.update(loss.item(), inputs.size(0))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        logging.info('Epoch {} Train, Loss: {:.5f}, lr: {:.5f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), self.optimizer.param_groups[0]['lr'], time.time()-epoch_start))
        
        self.tr_graph(self.epoch, [epoch_loss.get_avg()])

        if epoch % self.args.check_point == 0:
            model_state_dic = self.model.state_dict()
            save_path = os.path.join(self.save_dir, 'cv_' + str(i),  '{}_ckpt.tar'.format(self.epoch))
            torch.save({
                'epoch': self.epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_state_dict': model_state_dic
            }, save_path)
            self.save_list.append(save_path)  # control the number of saved models

    def val_epoch(self, epoch, i):
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_loss = AverageMeter()

        for inputs, target in tqdm(self.val_loader, ncols=60):
            inputs = inputs.to(self.device)
            target = target.to(self.device)

            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, target)

            epoch_loss.update(loss.item(), inputs.size(0))

        logging.info('Epoch {} Val, Loss: {:.5f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), time.time()-epoch_start))

        self.vl_graph(self.epoch, [epoch_loss.get_avg()])

        model_state_dic = self.model.state_dict()
        if self.best_loss > epoch_loss.get_avg():
            self.best_loss = epoch_loss.get_avg()
            logging.info("save min loss {:.2f} model epoch {}".format(self.best_loss, self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'cv_' + str(i), 'best_model.pth'))