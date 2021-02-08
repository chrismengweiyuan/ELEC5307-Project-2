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
import torch.optim as optim
import torchvision.models as models


def parse_args():
    parser = argparse.ArgumentParser(description= \
        'scipt for part 3 of project 1')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='Used when there are cuda installed.')
    parser.add_argument('--output_path', default='./', type=str,
        help='The path that stores the log files.')
    parser.add_argument('--pretrained', action='store_true', default=False,
        help='When using this option, only run the test functions.')

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


# training process. 
def train_net(net, trainloader, valloader, criterion, optimizer, scheduler, epochs=40):
    val_accuracy = 0
    best_accuracy = 0;
    for epoch in range(epochs):  # loop over the dataset multiple times, only 1 time by default
        print('epoch: ',epoch+1)
        if type(scheduler).__name__ != 'NoneType':
            scheduler.step()
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if args.cuda:
                loss = loss.cpu()

            # print statistics and write to log
            running_loss += loss.item()
            if i % 10 == 9:    
                logging.info('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0 
    
        # evaluate the network after each epochs
        net = net.eval()
        correct = 0
        total = 0
        for data in valloader:
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
        logging.info('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
        logging.info('=' * 55)
        val_accuracy = 100 * correct / total
        
        # save the best performance model
        if(best_accuracy < val_accuracy):
            # save network
            torch.save(net.state_dict(), args.output_path + 'part1.pth')
            best_accuracy = val_accuracy
            
    # write finish to the flie        
    logging.info('Finished Training')
    return val_accuracy


args = parse_args()
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)
# print('using args:\n', args)

logger = create_logger(args.output_path)
logger.info('using args:')
logger.info(args)

# Transformation definition

train_transform = transforms.Compose([
    transforms.RandomResizedCrop((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=(0.6,1.4), contrast=0.4, saturation=(0.6,1.4), hue=0),
    transforms.ToTensor(), 
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(), 
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Define the training dataset and dataloader.

train_image_path = '/home/ec2-user/SageMaker/train' 
validation_image_path = '/home/ec2-user/SageMaker/val' 

trainset = ImageFolder(train_image_path, train_transform)
valset = ImageFolder(validation_image_path, test_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                         shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=16,
                                         shuffle=True, num_workers=2)
classes = ('apple', 'avocado', 'banana', 'bluebarry', 'cherry', 'dragonfruit',
           'grape', 'kiwifruit', 'lemon', 'orange', 'papaya', 'peach',
           'pear', 'pineapple', 'plum', 'pomegranate', 'rockmelon', 'strawberry')

args = parse_args()
network = models.resnet50(pretrained=True)
network.fc = nn.Linear(2048,18)
#print(network)

if args.cuda:
    network = network.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=0.00015, momentum=0.9,nesterov=True) 
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = 40)
# train and eval the trained network
val_acc = train_net(network, trainloader, valloader, criterion, optimizer, scheduler)

print("final validation accuracy:", val_acc,"%")