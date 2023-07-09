from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torchvision import transforms
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import os
import json
import chardet


class Dataset(Dataset):
    def __init__(self, guids, texts, imgs, labels):
        self.guids = guids
        self.texts = texts
        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return len(self.guids)

    def __getitem__(self, index):
        return self.guids[index], self.texts[index], self.imgs[index], self.labels[index]

    @staticmethod
    def collate_fn(batch):
        guids = [b[0] for b in batch]
        texts = [torch.LongTensor(b[1]) for b in batch]
        imgs = torch.FloatTensor([np.array(b[2]).tolist() for b in batch])
        labels = torch.LongTensor([b[3] for b in batch])
        texts_mask = [torch.ones_like(text) for text in texts]
        texts_padding = pad_sequence(texts, batch_first=True, padding_value=0)
        texts_padding_mask = pad_sequence(texts_mask, batch_first=True, padding_value=0).gt(0)
        return guids, texts_padding, texts_padding_mask, imgs, labels


class Vocab:
    def __init__(self):
        self.label2id = {}
        self.id2label = {}

    def __len__(self):
        return len(self.label2id)

    def add_label(self, label):
        if label not in self.label2id:
            idx = len(self.label2id)
            self.label2id[label] = idx
            self.id2label[idx] = label

    def label_to_id(self, label):
        return self.label2id.get(label)

    def id_to_label(self, id):
        return self.id2label.get(id)


class Process:
    def __init__(self, config):
        self.config = config
        self.vocab = Vocab()

    def __call__(self, data, params):
        guids, texts, images, labels = self.encode(data, self.vocab, self.config)
        dataset_ = Dataset(guids, texts, images, labels)
        return DataLoader(dataset_, **params, collate_fn=dataset_.collate_fn)

    def encode(self, data, vocab, config):
        for label in ['positive', 'neutral', 'negative', 'null']:
            vocab.add_label(label)
        tokenizer = AutoTokenizer.from_pretrained(config.bert_name)
        img_transform = transforms.Compose([
            transforms.Resize(2 ** (config.image_size - 1).bit_length()),
            transforms.CenterCrop(config.image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        guids, encoded_texts, encoded_imgs, encoded_labels = [], [], [], []
        for line in data:
            guid, text, img, label = line
            guids.append(guid)
            text = text.replace('#', '')
            tokens = tokenizer.tokenize(f'[CLS]{text}[SEP]')
            encoded_texts.append(tokenizer.convert_tokens_to_ids(tokens))
            encoded_imgs.append(img_transform(img))
            encoded_labels.append(vocab.label_to_id(label))
        return guids, encoded_texts, encoded_imgs, encoded_labels

    def decode(self, outputs):
        outputs_formated = ['guid,tag']
        for guid, label in outputs:
            outputs_formated.append(f'{guid},{self.vocab.id_to_label(label)}')
        return outputs_formated

    def format(self, input_path, data_dir, output_path):
        data = []
        with open(input_path) as f:
            for line in f.readlines():
                guid, label = line.replace('\n', '').split(',')
                text_path = os.path.join(data_dir, (guid + '.txt'))
                if guid == 'guid':
                    continue
                with open(text_path, 'rb') as tf:
                    text_byte = tf.read()
                    try:
                        text = text_byte.decode(chardet.detect(text_byte)['encoding'])
                    except:
                        try:
                            text = text_byte.decode('iso-8859-1').encode('iso-8859-1').decode('gbk')
                        except:
                            continue
                text = text.strip('\n').strip('\r').strip(' ').strip()
                data.append({'guid': guid, 'label': label, 'text': text})
        with open(output_path, 'w') as wf:
            json.dump(data, wf, indent=4)