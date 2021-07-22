import os
import logging
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from torchvision import transforms

from models.resnet import MultiBranchResNet
from models.efficient_net import MultiBranchEfficientNet,
from models.fusion_net import DoubleNet
from utils.helper import fix_seed

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train SimSiam')
    parser.add_argument('--exp-name', default='exp02',
                        help='exp results file name')
    parser.add_argument('--data-dir', default='',
                        help='training data directory')
    parser.add_argument('--save-dir', default='logs_test',
                        help='directory to save models.')
              
    parser.add_argument('--arch', type=str, default='resnet34',
                        help='the model architecture [resnet18, resnet34]')
    parser.add_argument('--res-dir', type=str, default='',
                        help='train result dir')

    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--crop-size', type=int, default=256,
                        help='the crop size of the train image')             

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip('-')  # set vis gpu

    fix_seed(765)

    sub_dir = datetime.strftime(datetime.now(), '%m%d-%H%M%S')  # prepare saving path
    sub_dir = '{}-'.format(args.exp_name) + sub_dir

    save_dir = os.path.join(args.save_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_dir = args.data_dir
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_count = torch.cuda.device_count()
        logging.info('using {} gpus'.format(device_count))
    else:
        raise Exception("gpu is not available")

    test_csv_path = os.path.join(data_dir, 'test.csv')
    test_df = pd.read_csv(test_csv_path)

    test_transforms = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model_path_list = os.listdir(args.res_dir)
    model_path_list.remove('args.json')
    model_path_list.remove('train.log')
    print(model_path_list)

    submission = pd.DataFrame()
    res = np.zeros((len(test_df)), dtype=np.float32)
    for model_path in model_path_list:
        cv_res = []

        if 'resnet' in args.arch:
            model = MultiBranchResNet(arch=args.arch, num_classes=1, mate_classes=5 + 1, tech_classes=2 + 1)
        elif 'efficientnet' in args.arch:
            model = MultiBranchEfficientNet(arch=args.arch, num_classes=1, mate_classes=5 + 1, tech_classes=2 + 1)
        elif 'fusion' in args.arch:
            model = DoubleNet()

        checkpoit = torch.load(os.path.join(args.res_dir, model_path, 'best_model.pth'))
        model.load_state_dict(checkpoit)
        model.to(device)

        model.eval()
        for i, test in enumerate(tqdm(test_df['object_id'], ncols=60)):
            img_path = os.path.join(data_dir, 'photos', test + '.jpg')
            img = Image.open(img_path)
            img = test_transforms(img)
            img.unsqueeze_(0)

            img = img.to(device)
            with torch.no_grad():
                output = model(img)
            output, _, _ = output

            out = output.item()
            cv_res.append(out)

        res += np.array(cv_res)

    submission['target'] = list(np.squeeze(res / len(model_path_list)))
    submission['target'].to_csv(os.path.join(save_dir, 'submission_{}.csv'.format(args.exp_name)), index=False)