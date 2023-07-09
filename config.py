import os

class Config:
    def __init__(self):
        self.root_path = os.getcwd()
        self.data_dir = os.path.join(self.root_path, 'data/')
        self.train_data_path = os.path.join(self.data_dir, 'train.json')
        self.test_data_path = os.path.join(self.data_dir, 'test.json')
        self.output_test_path = os.path.join(self.root_path, 'test.txt')
        self.output_path = os.path.join(self.root_path, 'checkpoint/')
        self.load_model_path = os.path.join(self.root_path, 'checkpoint/')

        self.set_dataloader_params()
        self.set_common_params()
        self.set_fusion_params()
        self.set_bert_params()
        self.set_resnet_params()

    def set_dataloader_params(self):
        self.train_params = {'batch_size': 16, 'shuffle': True, 'num_workers': 2}
        self.val_params = {'batch_size': 16, 'shuffle': False, 'num_workers': 2}
        self.test_params =  {'batch_size': 8, 'shuffle': False, 'num_workers': 2}

    def set_common_params(self):
        self.epoch = 20
        self.learning_rate = 3e-5
        self.weight_decay = 0
        self.num_labels = 3

    def set_fusion_params(self):
        self.model = 'model'
        self.only = None
        self.middle_hidden_size = 64
        self.contrastive_loss_weight = 0
        self.temperature = 0.5
        self.attention_nhead = 16
        self.attention_dropout = 0.4
        self.fuse_dropout = 0.5
        self.out_hidden_size = 128

    def set_bert_params(self):
        self.bert_name = 'xlm-roberta-base'
        self.bert_dropout = 0.2

    def set_resnet_params(self):
        self.image_size = 224
        self.resnet_dropout = 0.2