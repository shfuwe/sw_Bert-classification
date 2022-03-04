import torch
import jieba
from train99 import transformers_bert_binary_classification


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH = 'D:/Data/Linux/Suwen/Bert-classification/result/classifier9'
classifier= torch.load(PATH,map_location=torch.device('cpu'))
classifier.device=torch.device('cpu')
# classifier.device_setup()
ty = ['ICT', '新能源汽车', '生物医药', '医疗器械', '钢铁', '能源', '工业机器人', '先进轨道交通', '其他']
# print(classifier1.predict("『巴西』圣保罗城际铁路听证会延期至10月15日"))  # 0
# print(classifier1.predict("永恒力叉车入驻京东工业品 载重2吨的叉车设备也能线上采购"))  # 0

def read_list(text_path):
    lsit = []
    with open('%s' % text_path, 'r', encoding="utf8") as f:  # 打开一个文件只读模式
        line = f.readlines()  # 读取文件中的每一行，放入line列表中
        for line_list in line:
            lsit.append(line_list.replace('\n', ''))
    return lsit


def test():
    test_list = read_list('D:/Data/Linux/Suwen/Bert-classification/src/test.txt')
    for i in test_list:
        re = classifier.predict(i)  # 0
        result1 = re[1]
        result2 = re[0].tolist()

        if result2[0][result1] < 3.5:
            print(i, '\n', result1, '***** 其他 ***** 原预测:',ty[result1], result2[0][result1], '\n', result2[0], '\n')
        else:
            print(i, '\n', result1, ty[result1], result2[0][result1], '\n', result2[0], '\n')

test()