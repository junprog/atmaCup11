# code refer to https://github.com/ZhihengCV/Bayesian-Crowd-Counting/blob/c81c45d50405c36cdcd339006876a04faa742373/utils/trainer.py
import os
import json
import logging
from datetime import datetime

from utils.helper import setlogger

class Trainer(object):
    def __init__(self, args):
        sub_dir = datetime.strftime(datetime.now(), '%m%d-%H%M%S')  # prepare saving path
        sub_dir = '{}-'.format(args.exp_name) + sub_dir

        self.save_dir = os.path.join(args.save_dir, sub_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        setlogger(os.path.join(self.save_dir, 'train.log'))  # set logger

        with open(os.path.join(self.save_dir, 'args.json'), 'w') as opt_file:
            json.dump(vars(args), opt_file)

        for k, v in args.__dict__.items():  # save args
            logging.info("{}: {}".format(k, v))
            
        self.args = args

    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        raise NotImplementedError

    def train(self):
        """training one epoch"""
        raise NotImplementedError