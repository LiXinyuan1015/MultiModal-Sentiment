import torch
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score

class Trainer():
    def __init__(self, config, processor, model, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        self.config = config
        self.processor = processor
        self.model = model.to(device)
        self.device = device
        self.optimizer = self._create_optimizer()

    def _create_optimizer(self):
        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        def get_weight_decay_params(named_parameters):
            return [{'params': [p for n, p in named_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': self.config.weight_decay}, {'params': [p for n, p in named_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        params = get_weight_decay_params(self.model.bert.named_parameters())
        params.append({'params': other_params, 'lr': self.config.learning_rate, 'weight_decay': self.config.weight_decay})
        return AdamW(params, lr=self.config.learning_rate)

    def process_batch(self, batch):
        guids, texts, texts_mask, imgs, labels = batch
        texts = texts.to(self.device)
        texts_mask = texts_mask.to(self.device)
        imgs = imgs.to(self.device)
        labels = labels.to(self.device) if labels is not None else None
        return texts, texts_mask, imgs, labels

    def train(self, train_loader):
        self.model.train()
        loss_list = []
        true_labels, pred_labels = [], []
        for batch in tqdm(train_loader, desc='[Training] '):
            texts, texts_mask, imgs, labels = self.process_batch(batch)
            pred, loss = self.model(texts, texts_mask, imgs, labels=labels)
            loss_list.append(loss.item())
            true_labels.extend(labels.tolist())
            pred_labels.extend(pred.tolist())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        train_loss = round(sum(loss_list) / len(loss_list), 5)
        return train_loss, loss_list

    def valid(self, val_loader):
        self.model.eval()
        val_loss = 0
        true_labels, pred_labels = [], []
        for batch in tqdm(val_loader, desc='[Validating] '):
            texts, texts_mask, imgs, labels = self.process_batch(batch)
            pred, loss = self.model(texts, texts_mask, imgs, labels=labels)
            val_loss += loss.item()
            true_labels.extend(labels.tolist())
            pred_labels.extend(pred.tolist())
        print(classification_report(true_labels, pred_labels))
        metrics = accuracy_score(true_labels, pred_labels)
        return val_loss / len(val_loader), metrics

    def predict(self, test_loader):
        self.model.eval()
        pred_guids, pred_labels = [], []
        for batch in tqdm(test_loader, desc='[Predicting] '):
            guids, texts, texts_mask, imgs, _ = batch
            texts, texts_mask, imgs, _ = self.process_batch([guids, texts, texts_mask, imgs, None])
            pred = self.model(texts, texts_mask, imgs, labels=None)
            pred_guids.extend(guids)
            pred_labels.extend(pred.tolist())
        return [(guid, label) for guid, label in zip(pred_guids, pred_labels)]