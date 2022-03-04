import torch
import jieba
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import AdamW
from transformers import BertModel,BertTokenizer
www=[2.980574324324324, 1.0, 4.089223638470452, 9.748618784530386, 7.02988047808765, 2.204247345409119, 3.992081447963801, 8.889168765743072, 8.503614457831326, 11.80267558528428, 3.529, 4.433417085427136, 3.529, 3.529]





    
"""
手动实现transformer.models.bert.BertForSequenceClassification()函数
根据论文[How to Fine-Tune BERT for Text Classification（2019）](https://www.aclweb.org/anthology/P18-1031.pdf)
在分类问题上，把最后四层进行concat然后maxpooling 输出的结果会比直接输出最后一层的要好
这里进行实现测试

"""

class bert_lr_last4layer_Config(nn.Module):
    def __init__(self):
        self.bert_path = "../chinese-bert-wwm"
        self.config_path = "../chinese-bert-wwm/config.json"

        # self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.num_labels = type_num
        # self.dropout_bertout = 0.2
        self.dropout_bertout = 0.5
        self.mytrainedmodel = "../result/bert_clf_model.bin"
        """
        current loss: 0.4363991916179657 	 current acc: 0.8125
        current loss: 0.1328232882924341 	 current acc: 0.9527363184079602
        current loss: 0.11797185830000853 	 current acc: 0.9585411471321695
        train loss:  0.11880445411248554 	 train acc: 0.9583704495516361
        valid loss:  0.1511497257672476 	 valid acc: 0.9431549028896258
        """

class bert_lr_last4layer(nn.Module):

    def __init__(self,config):
        super(bert_lr_last4layer, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path,config = config.config_path)
        self.dropout_bertout = nn.Dropout(config.dropout_bertout)
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.sigmoid=nn.Sigmoid() 
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=True,
        return_dict=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # outputs = outputs[2] # [1]是pooled的结果 # [3]是hidden_states 12层
        hidden_states = outputs.hidden_states
        nopooled_output = torch.cat((hidden_states[9],hidden_states[10],hidden_states[11],hidden_states[12]),1)
        batch_size = nopooled_output.shape[0] # 32
        # print(batch_size)
        # print(nopooled_output.shape) # torch.Size([32, 400, 768])
        kernel_hight = nopooled_output.shape[1]
        pooled_output = F.max_pool2d(nopooled_output,kernel_size = (kernel_hight,1))
        # print(pooled_output.shape) # torch.Size([32, 1, 768])

        flatten = pooled_output.view(batch_size,-1)
        # print(flatten.shape) # [32,768]

        flattened_output = self.dropout_bertout(flatten)

        logits = self.classifier(flattened_output)
        logits=self.sigmoid(logits)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return loss,logits


class Config(object):
    def __init__(self):
        self.config_dict = {
            "data_path": {
                # "trainingSet_path": "../data/sentiment/sentiment.train0.data",
                # "valSet_path": "../data/sentiment/sentiment.valid0.data",
                "trainingSet_path": trainingSet_path0,
                "valSet_path": valSet_path0,
                "testingSet_path": "../data/sentiment/sentiment.test0.data",
                "zeng_path": zeng_path0,
                "bad_path":bad_path0
            },

            "BERT_path": {
                "file_path": '../chinese-bert-wwm/',
                "config_path": '../chinese-bert-wwm/',
                "vocab_path": '../chinese-bert-wwm/',
            },

            "training_rule": {
                "max_length": 100,  # 输入序列长度，别超过512
                "hidden_dropout_prob": 0.3,
                "num_labels": type_num,  # 几分类个数
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

class SentimentDataset(Dataset):
    def __init__(self, path_to_file):
#         print(path_to_file)
        self.dataset = pd.read_csv(path_to_file, sep="\t", names=["text", "label"])
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        text = self.dataset.loc[idx, "text"]
        label = self.dataset.loc[idx, "label"]
        sample = {"text": text, "label": label}
        # print(sample)
        return sample

def convert_text_to_ids(tokenizer, text, max_len=100):
    if isinstance(text, str):
        tokenized_text = tokenizer.encode_plus(text, max_length=max_len, add_special_tokens=True, truncation=True)
        input_ids = tokenized_text["input_ids"]
        token_type_ids = tokenized_text["token_type_ids"]
    elif isinstance(text, list):
        input_ids = []
        token_type_ids = []
        for t in text:
            tokenized_text = tokenizer.encode_plus(t, max_length=max_len, add_special_tokens=True, truncation=True)
            input_ids.append(tokenized_text["input_ids"])
            token_type_ids.append(tokenized_text["token_type_ids"])
    else:
        print("Unexpected input")
    return input_ids, token_type_ids

def seq_padding(tokenizer, X):
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
    if len(X) <= 1:
        return torch.tensor(X)
    L = [len(x) for x in X]
    ML = max(L)
    X = torch.Tensor([x + [pad_id] * (ML - len(x)) if len(x) < ML else x for x in X])
    return X
        
class transformers_bert_binary_classification(object):
    def __init__(self):
        self.config = Config()
        self.device_setup()
        self.sigmoid=nn.Sigmoid()

    def device_setup(self):
        """
        设备配置并加载BERT模型
        :return:
        """
        self.freezeSeed()
        # 使用GPU，通过model.to(device)的方式使用
        device_s = "cuda:" + cuda_num
        self.device = torch.device(device_s if torch.cuda.is_available() else "cpu")

        # import os
        # result_dir = "../result"
        # MODEL_PATH = self.config.get("BERT_path", "file_path")
        # config_PATH = self.config.get("BERT_path", "config_path")
        vocab_PATH = self.config.get("BERT_path", "vocab_path")

        # num_labels = self.config.get("training_rule", "num_labels")
        # hidden_dropout_prob = self.config.get("training_rule", "hidden_dropout_prob")

        # 通过词典导入分词器
        self.tokenizer = transformers.BertTokenizer.from_pretrained(vocab_PATH)
        # self.model_config = BertConfig.from_pretrained(config_PATH, num_labels=num_labels,
        #                                                hidden_dropout_prob=hidden_dropout_prob)
        # self.model = BertForSequenceClassification.from_pretrained(MODEL_PATH, config=self.model_config)
        """
        train loss:  0.10704718510208534 	 train acc: 0.9637151849872321
        valid loss:  0.17820182011222863 	 valid acc: 0.9459971577451445
        """
        # 如果想换模型，换成下边这句子
        # bert+lr 跟官方方法差不都
        # self.model = bert_lr(bert_lr_Config())
        # self.model = bert_cnn(bert_cnn_Config())
        self.model = bert_lr_last4layer(bert_lr_last4layer_Config())

        self.model.to(self.device)

    def model_setup(self, zeng=0):
        weight_decay = self.config.get("training_rule", "weight_decay")
        learning_rate = self.config.get("training_rule", "learning_rate")
        print("**model_setup:")
        print("zeng",zeng)
        if zeng == 1:
            learning_rate = learning_rate * 2
        # 定义优化器和损失函数
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(www)).float())
        self.criterion.to(self.device)

    def get_data(self):
        """
        读取数据
        :return:
        """
        train_set_path = self.config.get("data_path", "trainingSet_path")
        valid_set_path = self.config.get("data_path", "valSet_path")
        batch_size = self.config.get("training_rule", "batch_size")
        zeng_set_path = self.config.get("data_path", "zeng_path")
        bad_set_path=self.config.get("data_path", "bad_path")
        print(train_set_path,valid_set_path,batch_size,zeng_set_path,bad_set_path)

        # 数据读入
        # 加载数据集
        sentiment_train_set = SentimentDataset(train_set_path)
        sentiment_train_loader = DataLoader(sentiment_train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        sentiment_valid_set = SentimentDataset(valid_set_path)
        sentiment_valid_loader = DataLoader(sentiment_valid_set, batch_size=batch_size, shuffle=False, num_workers=2)

        sentiment_zeng_set = SentimentDataset(zeng_set_path)
        sentiment_zeng_loader = DataLoader(sentiment_zeng_set, batch_size=batch_size, shuffle=True, num_workers=2)
        
        sentiment_bad_set = SentimentDataset(bad_set_path)
        sentiment_bad_loader = DataLoader(sentiment_bad_set, batch_size=batch_size, shuffle=False, num_workers=2)

        return sentiment_train_loader, sentiment_valid_loader, sentiment_zeng_loader,sentiment_bad_loader
    
    def get_bad_loss(self,iterator,loss_t=1):
        epoch_acc = 0
        
        for _, batch in enumerate(iterator):
            label = batch["label"]
            text = batch["text"]
#             print(text,label)
            label=[int(i) for i in label]
            label=torch.tensor(label)
#             print(text,label)

            input_ids, token_type_ids = convert_text_to_ids(self.tokenizer, text)
            input_ids = seq_padding(self.tokenizer, input_ids)
            token_type_ids = seq_padding(self.tokenizer, token_type_ids)
            label = label.unsqueeze(1)
            input_ids, token_type_ids, label = input_ids.long(), token_type_ids.long(), label.long()
            input_ids, token_type_ids, label = input_ids.to(self.device), token_type_ids.to(self.device), label.to(
                self.device)
            output = self.model(input_ids=input_ids, token_type_ids=token_type_ids, labels=label)
            # 更改了以下部分
            # y_pred_label = output[1].argmax(dim=1)
            y_pred_prob = output[1]
            if loss_t==1:
                loss=0
                for i in range(len(y_pred_prob)):
                    pre=y_pred_prob[i][label[i]]
    #                 print(text[i])
    #                 print(pre)

                    loss+=pre

                loss=loss/500
                loss.backward()
                self.optimizer.step()
                # 梯度清零
                self.optimizer.zero_grad()
            
#             loss = output[0]
            # loss = self.criterion(y_pred_prob.view(-1, 2), label.view(-1))
            with torch.no_grad():
                y_pred_label = y_pred_prob.argmax(dim=1)
                acc = ((y_pred_label == label.view(-1)).sum()).item()
                epoch_acc += acc

            
            

        return epoch_acc / len(iterator.dataset.dataset)
        
        
    def train_an_epoch(self, iterator,iterator_bad, zeng=0):
        print("**train_an_epoch")
        print("zeng",zeng)
        self.model_setup(zeng)
        epoch_loss = 0
        epoch_acc = 0

        for i, batch in enumerate(iterator):
            label = batch["label"]
            text = batch["text"]
#             print(label)
            input_ids, token_type_ids = convert_text_to_ids(self.tokenizer, text)
            input_ids = seq_padding(self.tokenizer, input_ids)
            token_type_ids = seq_padding(self.tokenizer, token_type_ids)
            # 标签形状为 (batch_size, 1)
            label = label.unsqueeze(1)
            # 需要 LongTensor
            input_ids, token_type_ids, label = input_ids.long(), token_type_ids.long(), label.long()
            # 迁移到GPU
            input_ids, token_type_ids, label = input_ids.to(self.device), token_type_ids.to(self.device), label.to(
                self.device)
            output = self.model(input_ids=input_ids, token_type_ids=token_type_ids, labels=label)  # 这里不需要labels
            # BertForSequenceClassification的输出loss和logits
            # BertModel原本的模型输出是last_hidden_state，pooler_output
            # bert_cnn的输出是[batch_size, num_class]
            # print(numpy.array(torch.tensor(output).cpu()).shape)

            y_pred_prob = output[1]

            # 计算loss
            # 这个 loss 和 output[0] 是一样的
            loss = self.criterion(y_pred_prob.view(-1, type_num), label.view(-1))  # 多分类改这里
            loss.backward()
            
            self.optimizer.step()
            # 梯度清零
            self.optimizer.zero_grad()
            
            if i % 120 == 1:
                print('bad_loss_backward')
                bad_acc=self.get_bad_loss(iterator_bad)
            
            with torch.no_grad():
                y_pred_label = y_pred_prob.argmax(dim=1)
                # 计算acc
                acc = ((y_pred_label == label.view(-1)).sum()).item()
                # epoch 中的 loss 和 acc 累加
                epoch_loss += loss.item()
                epoch_acc += acc
            if i % 120 == 0:
                bad_acc=self.get_bad_loss(iterator_bad,0)
                print('batch:',i,'/',len(iterator),"\tcurrent loss:", epoch_loss / (i + 1), "\tcurrent acc:", epoch_acc / ((i + 1) * len(label)),'\tbad_acc:',bad_acc)
        return epoch_loss / len(iterator), epoch_acc / len(iterator.dataset.dataset)

    def evaluate(self, iterator,xian):
        self.model.eval()
        epoch_loss = 0
        epoch_acc = 0
        y_pred_label_all = []
        label_all = []
        with torch.no_grad():
            for _, batch in enumerate(iterator):
                label = batch["label"]
                text = batch["text"]

                input_ids, token_type_ids = convert_text_to_ids(self.tokenizer, text)
                input_ids = seq_padding(self.tokenizer, input_ids)
                token_type_ids = seq_padding(self.tokenizer, token_type_ids)
                label = label.unsqueeze(1)
                input_ids, token_type_ids, label = input_ids.long(), token_type_ids.long(), label.long()
                input_ids, token_type_ids, label = input_ids.to(self.device), token_type_ids.to(self.device), label.to(
                    self.device)
                output = self.model(input_ids=input_ids, token_type_ids=token_type_ids, labels=label)
                # 更改了以下部分
                # y_pred_label = output[1].argmax(dim=1)
                y_pred_prob = output[1]
                
                y_pred_label=[]
                for i in y_pred_prob:
                    if max(i)>xian:
                        y_pred_label.append(i.argmax(dim=0))
                    else:
#                         print(i)
#                         print(max(i))
                        y_pred_label.append(torch.tensor(-1).to(self.device))
#                         print(y_pred_label)
#                 y_pred_label=np.array(y_pred_label)
                y_pred_label=torch.tensor(y_pred_label).to(self.device)
    
                loss = output[0]
                # loss = self.criterion(y_pred_prob.view(-1, 2), label.view(-1))
                acc = ((y_pred_label == label.view(-1)).sum()).item()
                y_pred_label_all += y_pred_label.tolist()
                label_all += label.view(-1).tolist()

                epoch_loss += loss.item()
                epoch_acc += acc
        target_names=['其他','ICT','新能源汽车','生物医药','医疗器械','钢铁','能源','工业机器人','先进轨道交通','数控机床','工业软件','高端装备','半导体','人工智能','稀土']
        print(metrics.classification_report(label_all, y_pred_label_all,target_names=target_names))
        print("准确率:", metrics.accuracy_score(label_all, y_pred_label_all))
        return epoch_loss / len(iterator), epoch_acc / len(iterator.dataset.dataset)
    
    def evval(self,xian):
        sentiment_train_loader, sentiment_valid_loader, sentiment_zeng_loader ,sentiment_bad_loader= self.get_data()
        valid_loss, valid_acc = self.evaluate(sentiment_valid_loader,xian)
        print("valid loss: ", valid_loss, "\t", "valid acc:", valid_acc)
    
    def evbad(self):
        sentiment_train_loader, sentiment_valid_loader, sentiment_zeng_loader ,sentiment_bad_loader= self.get_data()
        bad_acc=self.get_bad_loss(sentiment_bad_loader,0)
        print("bad_acc: ", bad_acc)
        
    def train(self, epochs, zeng=0):
        sentiment_train_loader, sentiment_valid_loader, sentiment_zeng_loader ,sentiment_bad_loader= self.get_data()
        best_valid_loss=999
        best_valid_acc=0
        for i in range(epochs):
            print('____________________________________________________________________________________')
            print('____________________________________________________________________________________')
            print('epochs:', i)
            print('____________________________________________________________________________________')
            print('____________________________________________________________________________________')
            print('____train____')
            if zeng == 0:
                train_loss, train_acc = self.train_an_epoch(sentiment_train_loader,sentiment_bad_loader)
            else:
                train_loss, train_acc = self.train_an_epoch(sentiment_zeng_loader, 1)
            print("train loss: ", train_loss, "\t", "train acc:", train_acc)
            print('____evaluate____')
            valid_loss, valid_acc = self.evaluate(sentiment_valid_loader,0.97)
            if valid_loss<best_valid_loss or valid_acc>best_valid_acc:
                best_valid_loss=min(best_valid_loss,valid_loss)
                best_valid_acc=max(best_valid_acc,valid_acc)
                torch.save(classifier, model_save_path+'_'+str(i)+'_'+str(round(valid_loss,5))+'_'+str(round(valid_acc,5)))
            print("valid loss: ", valid_loss, "\t", "valid acc:", valid_acc)
        # self.save_model()

    def save_model(self):
        model_save_path = self.config.get("result", "model_save_path")
        config_save_path = self.config.get("result", "config_save_path")
        vocab_save_path = self.config.get("result", "vocab_save_path")

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), model_save_path)
        # model_to_save.config.to_json_file(config_save_path) # !!!'bert_lr' object has no attribute 'config'
        # self.tokenizer.save_vocabulary(vocab_save_path)
        print("model saved...")

    def predict(self, sentence):
        # self.model.setup()
        self.model_setup()
        self.model.eval()
        # 转token后padding
        input_ids, token_type_ids = convert_text_to_ids(self.tokenizer, sentence)
        input_ids = seq_padding(self.tokenizer, [input_ids])
        token_type_ids = seq_padding(self.tokenizer, [token_type_ids])
        # 需要 LongTensor
        input_ids, token_type_ids = input_ids.long(), token_type_ids.long()
        # 梯度清零
        self.optimizer.zero_grad()
        # 迁移到GPU
        input_ids, token_type_ids = input_ids.to(self.device), token_type_ids.to(self.device)
        output = self.model(input_ids=input_ids, token_type_ids=token_type_ids)
        # y_pred_prob:各个类别的概率
        y_pred_prob = output[0]
#         y_pred_prob=self.sigmoid(y_pred_prob)
        # 取概率最大的标签
        y_pred_label = y_pred_prob.argmax(dim=1)

        # 将torch.tensor转换回int形式
        return y_pred_prob, y_pred_label.item()

    def freezeSeed(self):
        seed = 1
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        

def read_list(text_path):
    lsit = []
    with open('%s' % text_path, 'r', encoding="utf8") as f:  # 打开一个文件只读模式
        line = f.readlines()  # 读取文件中的每一行，放入line列表中
        for line_list in line:
            lsit.append(line_list.replace('\n', ''))
    return lsit


def test(test_list,classifier,ty):
    res=[]
    for i in test_list:
        re = classifier.predict(i)  # 0
        result1 = re[1]
        result2 = re[0].tolist()
        
        if result2[0][result1] < 0.997:
            res.append('其他')
            print(i, '\n', result1, '***** 其他 ***** 原预测:',ty[result1], result2[0][result1], '\n', result2[0], '\n')
        else:
            res.append(ty[result1])
            print(i, '\n', result1, ty[result1], result2[0][result1], '\n', result2[0], '\n')
        return res



def run(list_):
    device = torch.device("cpu")
    model_save_path='./classifier_12.13_8_1.88932_0.8347'
    classifier= torch.load(model_save_path,map_location=device)
    classifier.device=torch.device('cpu')
    # print(classifier1.predict("『巴西』圣保罗城际铁路听证会延期至10月15日"))  # 0
    # print(classifier1.predict("永恒力叉车入驻京东工业品 载重2吨的叉车设备也能线上采购"))  # 0
    ty = ['ICT', '新能源汽车', '生物医药', '医疗器械', '钢铁', 
       '能源', '工业机器人', '先进轨道交通', '数控机床', '工业软件', 
       '高端装备', '半导体', '人工智能', '稀土']
#     test_list = read_list('/data/fuwen/SuWen/Bert-classification/src/test.txt')
    res=test(list_,classifier,ty) #主要是这句
    
# run()