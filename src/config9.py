# -*- coding: utf-8 -*-

"""
Created on 2020-07-29 09:03
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com

配置模型、路径、与训练相关参数
"""


class Config(object):
    def __init__(self):
        self.config_dict = {
            "data_path": {
                # "trainingSet_path": "../data/sentiment/sentiment.train0.data",
                # "valSet_path": "../data/sentiment/sentiment.valid0.data",
                "trainingSet_path": "../data/sentiment/quan_9/train_10.14.txt",
                "valSet_path": "../data/sentiment/quan_9/test_10.14.txt",
                "testingSet_path": "../data/sentiment/sentiment.test0.data",
                "zeng_path": "../data/sentiment/quan_9/zeng.txt"
            },

            "BERT_path": {
                "file_path": '../chinese-bert-wwm/',
                "config_path": '../chinese-bert-wwm/',
                "vocab_path": '../chinese-bert-wwm/',
            },

            "training_rule": {
                "max_length": 300,  # 输入序列长度，别超过512
                "hidden_dropout_prob": 0.3,
                "num_labels": 9,  # 几分类个数
                "learning_rate": 1e-5,
                "weight_decay": 1e-2,
                "batch_size": 16
            },

            "result": {
                "model_save_path": '../result/bert_clf_model.bin',
                "config_save_path": '../result/bert_clf_config.json',
                "vocab_save_path": '../result/bert_clf_vocab.txt'
            }
        }

    def get(self, section, name):
        return self.config_dict[section][name]
