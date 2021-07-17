import os
import torch
import argparse

from engine.multi_task_trainer import MultiTaskTrainer

from utils.helper import fix_seed

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train SimSiam')
    parser.add_argument('--exp-name', default='exp04',
                        help='exp results file name')
    parser.add_argument('--data-dir', default='',
                        help='training data directory')
    parser.add_argument('--save-dir', default='logs_multitask',
                        help='directory to save models.')
              
    parser.add_argument('--arch', type=str, default='resnet34',
                        help='the model architecture [resnet18, resnet34, efficientnet_b0]')
    parser.add_argument('--init-weight-path', type=str, default='',
                        help='ssl pre-train model path')
                      
    parser.add_argument('--resume', default='',
                        help='the path of resume training model')
    parser.add_argument('--max-model-num', type=int, default=1,
                        help='max models num to save ')
    parser.add_argument('--check_point', type=int, default=100,
                        help='milestone of save model checkpoint')

    parser.add_argument('--max-epoch', type=int, default=401,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=5,
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=0,
                        help='the epoch start to val')

    parser.add_argument('--batch-size', type=int, default=128,
                        help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='the num of training process')

    parser.add_argument('--crop-size', type=int, default=256,
                        help='the crop size of the train image')          

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip('-')  # set vis gpu

    fix_seed(765)

    trainer = MultiTaskTrainer(args)
    trainer.setup()
    trainer.train()
