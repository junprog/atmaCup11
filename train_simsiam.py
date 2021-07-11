
import argparse
import os
import math
import torch

from engine.simsiam_trainer import SimSiamTrainer

from utils.helper import fix_seed

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train SimSiam')
    parser.add_argument('--exp_name', default='exp02',
                        help='exp results file name')
    parser.add_argument('--data-dir', default='',
                        help='training data directory')
    parser.add_argument('--save-dir', default='simsiam_logs',
                        help='directory to save models.')
              
    parser.add_argument('--arch', type=str, default='resnet34',
                        help='the model architecture [resnet18, resnet34]')
                      
                      
    parser.add_argument('--resume', default='',
                        help='the path of resume training model')
    parser.add_argument('--max-model-num', type=int, default=30,
                        help='max models num to save ')
    parser.add_argument('--check_point', type=int, default=50,
                        help='milestone of save model checkpoint')

    parser.add_argument('--max-epoch', type=int, default=301,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=10,
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=0,
                        help='the epoch start to val')

    parser.add_argument('--batch-size', type=int, default=8,
                        help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='the num of training process')

    parser.add_argument('--crop-size', type=int, default=224,
                        help='the crop size of the train image')

    parser.add_argument('--visual-num', type=int, default=4,
                        help='the number of visualize images')                       

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip('-')  # set vis gpu

    fix_seed(765)

    trainer = SimSiamTrainer(args)
    trainer.setup()
    trainer.train()
