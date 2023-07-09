import torch
import torch.nn as nn
from transformers import AutoModel

class RobertaModel(nn.Module):
    def __init__(self, config):
        super(RobertaModel, self).__init__()
        self.bert = AutoModel.from_pretrained(config.bert_name)
        self.classifier = nn.Sequential(
            nn.Dropout(config.bert_dropout),
            nn.Linear(self.bert.config.hidden_size, config.out_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.out_hidden_size, config.num_labels)
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, texts, texts_mask, imgs, labels=None):
        bert_out = self.bert(input_ids=texts, token_type_ids=None, attention_mask=texts_mask)
        pooled_output = bert_out['pooler_output']
        prob = self.classifier(pooled_output)
        pred_labels = torch.argmax(prob, dim=1)
        if labels is not None:
            loss = self.loss(prob, labels)
            return pred_labels, loss
        else:
            return pred_labels