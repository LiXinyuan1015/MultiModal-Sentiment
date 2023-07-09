import torch
import torch.nn as nn
from torchvision.models import resnet152
from transformers import AutoModel
from torch.nn.functional import normalize

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = AutoModel.from_pretrained(config.bert_name)
        self.text_model = nn.Sequential(
            nn.Dropout(config.bert_dropout),
            nn.Linear(self.bert.config.hidden_size, config.middle_hidden_size),
            nn.ReLU(inplace=True)
        )
        self.resnet152 = resnet152(pretrained=True)
        self.resnet = nn.Sequential(
            *(list(self.resnet152.children())[:-1]),
            nn.Flatten()
        )
        self.image_model = nn.Sequential(
            nn.Dropout(config.resnet_dropout),
            nn.Linear(self.resnet152.fc.in_features, config.middle_hidden_size),
            nn.ReLU(inplace=True)
        )
        self.attention = nn.TransformerEncoderLayer(
            d_model = config.middle_hidden_size * 2,
            nhead = config.attention_nhead,
            dropout = config.attention_dropout
        )
        self.classifier = nn.Sequential(
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.middle_hidden_size * 2, config.out_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.out_hidden_size, config.num_labels)
        )
        self.loss = nn.CrossEntropyLoss()
        self.contrastive_loss_weight = config.contrastive_loss_weight  # 对比损失权重

    def contrastive_loss(self, features, labels):
        features = normalize(features, dim=1)  # L2范数归一化特征
        similarity_matrix = torch.matmul(features, features.t())  # 计算相似度矩阵
        mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()  # 计算标签相等的掩码
        mask.fill_diagonal_(0)  # 将对角线填充为0，防止自身比较
        pos_mask = mask
        pos_sim = similarity_matrix * pos_mask
        neg_sim = similarity_matrix * (1 - pos_mask)
        max_sim, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        numerator = torch.exp((pos_sim - max_sim) / 0.5).sum(dim=1)
        denominator = torch.exp((similarity_matrix - max_sim) / 0.5).sum(dim=1)
        contrastive_loss = -torch.log(numerator / (denominator - numerator))
        return contrastive_loss.mean()

    def forward(self, texts, texts_mask, imgs, labels=None):
        bert_out = self.bert(input_ids=texts, token_type_ids=None, attention_mask=texts_mask)
        text_feature = self.text_model(bert_out['pooler_output'])
        img_feature = self.image_model(self.resnet(imgs))
        features = torch.cat([text_feature, img_feature], dim=1)  # 连接文本和图像特征
        attention_out = self.attention(features.unsqueeze(0)).squeeze()
        prob = self.classifier(attention_out)
        pred_labels = torch.argmax(prob, dim=1)

        if labels is not None:
            # 计算原始损失
            loss = self.loss(prob, labels)
            # 计算对比损失
            contrastive_loss = self.contrastive_loss(features, labels)
            # 将对比损失与原始损失相结合
            total_loss = loss + self.contrastive_loss_weight * contrastive_loss
            return pred_labels, total_loss
        else:
            return pred_labels