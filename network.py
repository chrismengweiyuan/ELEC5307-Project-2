import torch.nn as nn
import torchvision.models as models

#import the pretrained model
Network = models.resnet50(pretrained=False)
Network.fc = nn.Linear(2048,18)

