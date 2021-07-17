import os
import time
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import transforms

from engine.trainer import Trainer
from datasets.one_hot_encode import one_hot_encode
from datasets.atma_dataset import MultiAtmaDataset, MultiAtmaDataset_v2
from models.resnet import MultiBranchResNet, MultiBranchResNet_v2
from models.efficient_net import MultiBranchEfficientNet, MultiBranchEfficientNet_v2, MultiBranchEfficientNet_v3
from losses.multi_task_loss import MultiTaskLoss, MultiTaskLoss_v2, MultiTaskLoss_v3
from utils.helper import Save_Handle, AverageMeter
from utils.visualizer import GraphPlotter
from utils.metrics import calc_accuracy_ce

class MultiTaskTrainer(Trainer):
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
        self.img_path = os.path.join(self.data_dir, 'photos')

        material_path = os.path.join(self.data_dir, 'materials.csv')
        technique_path = os.path.join(self.data_dir, 'techniques.csv')

        self.train_df = pd.read_csv(train_csv_path)
        self.tech_df = pd.read_csv(technique_path)
        self.mate_df = pd.read_csv(material_path)
        
        self.encoded_mate_df = self.target_encoder(one_hot_encode(self.mate_df), self.mate_df, num=5)
        self.encoded_tech_df = self.target_encoder(one_hot_encode(self.tech_df), self.tech_df, num=2)
        
        # train, materials, techniques を紐付ける
        self.unit_mate_df = self.train_df.merge(self.encoded_mate_df, on='object_id', how='left')
        self.unit_mate_df.loc[self.unit_mate_df.paper.isnull(), 'other'] = 1
        self.unit_mate_df = self.unit_mate_df.fillna(0)

        self.unit_tech_df = self.train_df.merge(self.encoded_tech_df, on='object_id', how='left')
        self.unit_tech_df.loc[self.unit_tech_df.pen.isnull(), 'other'] = 1
        self.unit_tech_df = self.unit_tech_df.fillna(0)   

        assert len(self.unit_tech_df) == len(self.unit_mate_df), "mate, tech is not same size"

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
        #self.criterion = MultiTaskLoss(is_mse=False)
        #self.criterion = MultiTaskLoss_v2()
        self.criterion = MultiTaskLoss_v3()
        self.criterion.to(self.device)

        #self.val_criterion = nn.MSELoss()
        self.val_criterion = nn.CrossEntropyLoss()
        self.val_criterion.to(self.device)

        self.save_list = Save_Handle(max_num=args.max_model_num)

    def target_encoder(self, target, origin, num=2):
        """class: 10 -> 2 + other class 1 -> 3"""
        class_names = [class_name for class_name in origin.name.value_counts().index]
        use_class_names = class_names[:num]
        use_class_names.append('other')

        not_use_class_names = class_names[num:]

        # other making
        for name in not_use_class_names:
            target.loc[target[name] == 1, 'other'] = 1
            target = target.drop(name, axis=1)

        target = target.fillna(0)

        return target

    def train(self):
        """training process"""
        args = self.args

        for i, (train, val) in enumerate(self.skf.split(self.train_df['object_id'], self.train_df['target'])):

            if not os.path.exists(os.path.join(self.save_dir, 'cv_' + str(i))):
                os.mkdir(os.path.join(self.save_dir, 'cv_' + str(i)))

            self.tr_graph = GraphPlotter(os.path.join(self.save_dir, 'cv_' + str(i)), ['Loss'], 'train')
            self.vl_graph = GraphPlotter(os.path.join(self.save_dir, 'cv_' + str(i)), ['CE Loss', 'Acc'], 'val')

            # train_dataset = MultiAtmaDataset(
            #     data_dir = self.img_path,
            #     img_name_df = self.train_df.object_id[train],
            #     target_df = self.train_df.drop('object_id', axis=1).drop('sorting_date', axis=1).drop('art_series_id', axis=1).loc[train],
            #     mate_df = self.unit_mate_df.drop('object_id', axis=1).drop('sorting_date', axis=1).drop('art_series_id', axis=1).drop('target', axis=1).loc[train],
            #     tech_df = self.unit_tech_df.drop('object_id', axis=1).drop('sorting_date', axis=1).drop('art_series_id', axis=1).drop('target', axis=1).loc[train],
            #     trans = self.train_transforms,
            # )

            # val_dataset = MultiAtmaDataset(
            #     data_dir = self.img_path,
            #     img_name_df = self.train_df.object_id[val],
            #     target_df = self.train_df.drop('object_id', axis=1).drop('sorting_date', axis=1).drop('art_series_id', axis=1).loc[val],
            #     mate_df = self.unit_mate_df.drop('object_id', axis=1).drop('sorting_date', axis=1).drop('art_series_id', axis=1).drop('target', axis=1).loc[val],
            #     tech_df = self.unit_tech_df.drop('object_id', axis=1).drop('sorting_date', axis=1).drop('art_series_id', axis=1).drop('target', axis=1).loc[val],
            #     trans = self.val_transforms,
            # )

            train_dataset = MultiAtmaDataset_v2(
                data_dir = self.img_path,
                img_name_df = self.train_df.object_id[train],
                target_df = self.train_df.drop('object_id', axis=1).drop('sorting_date', axis=1).drop('art_series_id', axis=1).loc[train],
                year_df = self.train_df.drop('object_id', axis=1).drop('target', axis=1).drop('art_series_id', axis=1).loc[train],
                mate_df = self.unit_mate_df.drop('object_id', axis=1).drop('sorting_date', axis=1).drop('art_series_id', axis=1).drop('target', axis=1).loc[train],
                tech_df = self.unit_tech_df.drop('object_id', axis=1).drop('sorting_date', axis=1).drop('art_series_id', axis=1).drop('target', axis=1).loc[train],
                trans = self.train_transforms,
            )

            val_dataset = MultiAtmaDataset_v2(
                data_dir = self.img_path,
                img_name_df = self.train_df.object_id[val],
                target_df = self.train_df.drop('object_id', axis=1).drop('sorting_date', axis=1).drop('art_series_id', axis=1).loc[val],
                year_df = self.train_df.drop('object_id', axis=1).drop('target', axis=1).drop('art_series_id', axis=1).loc[val],
                mate_df = self.unit_mate_df.drop('object_id', axis=1).drop('sorting_date', axis=1).drop('art_series_id', axis=1).drop('target', axis=1).loc[val],
                tech_df = self.unit_tech_df.drop('object_id', axis=1).drop('sorting_date', axis=1).drop('art_series_id', axis=1).drop('target', axis=1).loc[val],
                trans = self.val_transforms,
            )

            self.train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
            self.val_loader = torch.utils.data.DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

            # Define model, scheduler, optim
            if 'resnet' in args.arch:
                #self.model = MultiBranchResNet_v2(arch=args.arch, num_classes=4, mate_classes=6, tech_classes=3)
                self.model = MultiBranchResNet(arch=args.arch, num_classes=1, mate_classes=6, tech_classes=3)
            elif 'efficientnet' in args.arch:
                self.model = MultiBranchEfficientNet_v3(arch=args.arch, num_classes=4, mate_classes=6, tech_classes=3)
                #self.model = MultiBranchEfficientNet_v2(arch=args.arch, num_classes=4, mate_classes=6, tech_classes=3)
                #self.model = MultiBranchEfficientNet(arch=args.arch, num_classes=1, mate_classes=6, tech_classes=3)

            if i == 0: 
                print(self.model)

            if args.init_weight_path:
                suf = args.init_weight_path.rsplit('.', 1)[-1]
                if suf == 'tar':
                    checkpoint = torch.load(args.init_weight_path)
                    new_checkpoint = OrderedDict()
                    
                    for saved_key, saved_value in checkpoint['model_state_dict'].items():
                        if 'projector' in saved_key or 'predictor' in saved_key or 'fc' in saved_key:
                            continue
                        else:
                            if 'resnet' in args.arch:
                                saved_key = saved_key.replace('encoder.', '')
                            elif 'efficientnet' in args.arch:
                                saved_key = saved_key.replace('encoder.0.', '')
                            new_checkpoint[saved_key] = saved_value

                    self.model.feature.load_state_dict(new_checkpoint)

                elif suf == 'pth':
                    checkpoint = torch.load(args.init_weight_path)
                    new_checkpoint = OrderedDict()
                    
                    for saved_key, saved_value in checkpoint.items():
                        if 'projector' in saved_key or 'predictor' in saved_key or 'fc' in saved_key:
                            continue
                        else:
                            if 'resnet' in args.arch:
                                saved_key = saved_key.replace('encoder.', '')
                            elif 'efficientnet' in args.arch:
                                saved_key = saved_key.replace('encoder.0.', '')
                            new_checkpoint[saved_key] = saved_value

                    self.model.feature.load_state_dict(new_checkpoint)
                
            self.model.to(self.device)

            lr = 0.01

            #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

            self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[500], gamma=0.1)
            #self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.max_epoch)

            self.start_epoch = 0
            self.best_loss = np.Inf
            self.best_acc = 0
            
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
        #if epoch == 0:
        #    for params in self.model.feature.parameters():
        #        params.requires_grad = False
        #elif epoch == 21:
        #    for params in self.model.feature.parameters():
        #        params.requires_grad = True

        for inputs, target, logit, target_soft, mate, tech in tqdm(self.train_loader, ncols=60):
        #for inputs, target, mate, tech in tqdm(self.train_loader, ncols=60):
            inputs = inputs.to(self.device)
            #target = target.to(self.device)
            logit = logit.to(self.device)
            target_soft = target_soft.to(self.device)
            mate = mate.to(self.device)
            tech = tech.to(self.device)

            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                #loss = self.criterion(outputs, target, mate, tech)
                #loss = self.criterion(outputs, target, logit, target_soft, mate, tech)
                loss = self.criterion(outputs, target_soft, logit, mate, tech)

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
        epoch_acc = AverageMeter()

        for inputs, target, logit, target_soft, mate, tech in tqdm(self.val_loader, ncols=60):
        #for inputs, target, mate, tech in tqdm(self.val_loader, ncols=60):
            inputs = inputs.to(self.device)
            #target = target.to(self.device)
            target = logit.to(self.device)

            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                #outputs, _, _ = outputs
                #outputs, _, _, _, _ = outputs
                _, outputs, _, _ = outputs
                loss = self.val_criterion(outputs, target.long())

            epoch_loss.update(loss.item(), inputs.size(0))
            epoch_acc.update(calc_accuracy_ce(outputs, target), inputs.size(0))

        logging.info('Epoch {} Val, Loss: {:.5f}, Acc: {:.2f} Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), epoch_acc.get_avg(), time.time()-epoch_start))

        self.vl_graph(self.epoch, [epoch_loss.get_avg(), epoch_acc.get_avg()])

        model_state_dic = self.model.state_dict()
        # if self.best_loss > epoch_loss.get_avg():
        #     self.best_loss = epoch_loss.get_avg()
        #     logging.info("save min loss {:.2f} model epoch {}".format(self.best_loss, self.epoch))
        #     torch.save(model_state_dic, os.path.join(self.save_dir, 'cv_' + str(i), 'best_model.pth'))
        if self.best_acc < epoch_acc.get_avg():
            self.best_acc = epoch_acc.get_avg()
            logging.info("save max acc {:.2f} model epoch {}".format(self.best_acc, self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'cv_' + str(i), 'best_model.pth'))