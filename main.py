import os
import torch
import argparse
from config import Config
from utils import Process
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
import chardet
from PIL import Image

cfg = Config()
data_processor = Process(cfg)
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--do_train', default=True, action='store_true', help='train model')
arg_parser.add_argument('--do_test', default=True, action='store_true', help='predict test data')
arg_parser.add_argument('--text_only', default=False, action='store_true', help='only use text')
arg_parser.add_argument('--img_only', default=False, action='store_true', help='only use image')
arg_parser.add_argument('--model', default='model', help='model, roberta(baseline), resnet(baseline)', type=str)
arg_parser.add_argument('--load_model_path', default="checkpoint/model/pytorch_model.bin", help='load model', type=str)
args = arg_parser.parse_args()

if cfg.model == 'model':
    from Trainer.trainer import Trainer
    from model.model import Model
elif cfg.model == 'roberta':
    from Trainer.trainer_roberta import Trainer
    from model.roberta import RobertaModel as Model
else:
    from Trainer.trainer_resnet import Trainer
    from model.resnet import ResNet as Model
cfg.load_model_path = args.load_model_path
cfg.model = args.model
if args.text_only:
    cfg.only = 'text'
if args.img_only:
    cfg.only = 'img'
if args.img_only and args.text_only:
    cfg.only = None

model = Model(cfg)
device_type = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
trainer = Trainer(cfg, data_processor, model, device_type)

def train():
    data_processor.format(os.path.join(cfg.root_path, 'data/train.txt'), os.path.join(cfg.root_path, 'data/'), os.path.join(cfg.root_path, 'data/train.json'))
    dataset = []
    with open(cfg.train_data_path) as f:
        json_data = json.load(f)
        for entry in tqdm(json_data, desc='[Loading]'):
            guid, label, text = entry['guid'], entry['label'], entry['text']
            if guid == 'guid':
                continue
            if cfg.only == 'text':
                 img = Image.new(mode='RGB', size=(224, 224), color=(0, 0, 0))
            else:
                img_path = os.path.join(cfg.data_dir, (guid + '.jpg'))
                img = Image.open(img_path)
                img.load()
            if cfg.only == 'img':
                text = ''
            dataset.append((guid, text, img, label))
        f.close()

    train_dataset, val_dataset = train_test_split(dataset, train_size=0.8, test_size=0.2)
    train_data_loader = data_processor(train_dataset, cfg.train_params)
    val_data_loader = data_processor(val_dataset, cfg.val_params)
    best_acc = 0
    for e in range(cfg.epoch):
        print('Epoch ' + str(e+1))
        train_loss, train_loss_list = trainer.train(train_data_loader)
        print('Train Loss: {}'.format(train_loss))
        validation_loss, val_acc = trainer.valid(val_data_loader)
        print('Valid Loss: {}'.format(validation_loss))
        print('Valid Acc: {}'.format(val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            output_model_directory = os.path.join(cfg.output_path, cfg.model)
            if not os.path.exists(output_model_directory):
                os.makedirs(output_model_directory)
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(output_model_directory, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
            print('Save best model')

def test():
    data_processor.format(os.path.join(cfg.root_path, 'data/test_without_label.txt'), os.path.join(cfg.root_path, 'data/'), os.path.join(cfg.root_path, 'data/test.json'))
    test_dataset = []
    with open(cfg.test_data_path) as f:
        json_data = json.load(f)
        for entry in tqdm(json_data, desc='[Loading]'):
            guid, label, text = entry['guid'], entry['label'], entry['text']
            if guid == 'guid': 
                continue
            if cfg.only == 'text': 
                img = Image.new(mode='RGB', size=(224, 224), color=(0, 0, 0))
            else:
                img_path = os.path.join(cfg.data_dir, (guid + '.jpg'))
                img = Image.open(img_path)
                img.load()
            if cfg.only == 'img': 
                text = ''
            test_dataset.append((guid, text, img, label))
        f.close()
    test_data_loader = data_processor(test_dataset, cfg.test_params)
    if cfg.load_model_path is not None:
        model.load_state_dict(torch.load(cfg.load_model_path))

    predicted_outputs = trainer.predict(test_data_loader)
    formatted_outputs = data_processor.decode(predicted_outputs)
    with open(cfg.output_test_path, 'w') as f:
        for line in tqdm(formatted_outputs, desc='[Writing]'):
            f.write(line)
            f.write('\n')
        f.close()

if __name__ == "__main__":
    if args.do_train:
        train()
    if args.do_test:
        if args.load_model_path is None and not args.do_train:
            print('no model to test')
        else:
            test()