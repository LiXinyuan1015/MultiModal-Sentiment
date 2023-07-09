import torch
import torch.nn as nn
from torchvision.models import resnet152

class ResNet(nn.Module):
    def __init__(self, config):
        super(ResNet, self).__init__()
        self.resnet152 = resnet152(pretrained=True)
        self.resnet = nn.Sequential(
            *(list(self.resnet152.children())[:-1]), # 忽略全连接层
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(config.resnet_dropout),
            nn.Linear(self.resnet152.fc.in_features, config.out_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.resnet_dropout),
            nn.Linear(config.out_hidden_size, config.num_labels)
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, texts, texts_mask, imgs, labels=None):
        img_feature = self.resnet(imgs)
        prob = self.classifier(img_feature)
        pred_labels = torch.argmax(prob, dim=1)
        if labels is not None:
            loss = self.loss(prob, labels)
            return pred_labels, loss
        else:
            return pred_labels