# import the packages
import argparse
import logging
import sys
import time
import os

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from network import Network # the network you used

# control input options.
def parse_args():
    parser = argparse.ArgumentParser(description= \
        'scipt for evaluation of project 2')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='Used when there are cuda installed.')
    parser.add_argument('--da', action='store_true', default=False,
        help='Used for evaluating domain adaptation task model.')
    parser.add_argument('--output_path', default='./', type=str,
        help='The path that stores the log files.')

    pargs = parser.parse_args()
    return pargs

# Creat logs. 
def create_logger(final_output_path):
    log_file = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(final_output_path, log_file),
                        format=head)
    clogger = logging.getLogger()
    clogger.setLevel(logging.INFO)
    # add handler
    # print to stdout and log file
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    clogger.addHandler(ch)
    return clogger

# evaluation process. 
def eval_net(net, loader, logging):
    net = net.eval()
    if args.cuda:
        net = net.cuda()

    if args.da:
        assert os.path.exists(args.output_path + 'project2_da.pth'), 'InputError: There is no pretrained file project2.pth. Please try again.'
    else:
        assert os.path.exists(args.output_path + 'project2.pth'), 'InputError: There is no pretrained file project2.pth. Please try again.'

    if args.da:
        model_name = args.output_path + 'project2_da.pth'
    else:
        model_name = args.output_path + 'project2.pth'

    if args.cuda:
        net.load_state_dict(torch.load(model_name, map_location='cuda'))
    else:
        net.load_state_dict(torch.load(model_name, map_location='cpu'))

    correct = 0
    total = 0
    for data in loader:
        images, labels = data
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        if args.cuda:
            outputs = outputs.cpu()
            labels = labels.cpu()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # print and write to log
    logging.info('=' * 55)
    logging.info('SUMMARY of Project2')
    logging.info('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
    logging.info('=' * 55)

# Prepare for writing logs and setting GPU. 
args = parse_args()
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)
# print('using args:\n', args)

logger = create_logger(args.output_path)
logger.info('using args:')
logger.info(args)

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(), 
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


# Define the test dataset and dataloader.

if args.da:
    test_image_path = '../target' 
else:
    test_image_path = '../test' 
testset = ImageFolder(test_image_path, test_transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=2)

network = Network
if args.cuda:
    network = network.cuda()

eval_net(network, testloader, logging)
