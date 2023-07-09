# 多模态情感分析

#### Setup

* torch==1.10.1
* tqdm==4.64.1
* transformers==4.30.2
* scikit-learn==1.0.2
* numpy==1.21.5
* argparse==1.4.0
* torchvision==0.11.2
* chardet==4.0.0

You can simply run

```
pip install -r requirements.txt
```

## Repository structure

```
code:.
│  config.py #参数配置文件
│  main.py #主程序
│  README.md
│  requirements.txt
│  stat.ipynb #统计作图
│  utils.py #数据处理方法
│
├─checkpoint #存放预训练模型及训练好的模型
│  ├─Roberta
│  └─XLMRoberta
├─data
│      test.json
│      test_without_label.txt
│      train.json
│      train.txt
│
├─model
│      model.py #多模态融合模型
│      resnet.py #baseline
│      roberta.py #baseline
│      __init__.py
│
└─Trainer
        trainer.py #多模态融合模型训练器
        trainer_resnet.py #baseline训练器
        trainer_roberta.py #baseline训练器
        __init__.py
```

## Run

进入code文件夹，需要将数据集中所有的jpg与txt文件放入data/路径下，通过以下指令运行代码：

```
python3 main.py
```

此外还可以通过添加参数运行不同模型：

* --model可以选择model, roberta(baseline), resnet(baseline)，分别为主要模型和两个基准模型
* 配置--text_only True或--img_only True可以进行消融实验
* --do_train True或--do_test True可以选择进行模型训练还是用已有模型预测，已有的训练好的模型会保存在checkpoint中
