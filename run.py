import torch 
from tqdm import tqdm
import torch.nn as nn 
from torch.optim import Adam
import numpy as np
import os,sys
sys.path.append(os.pardir)
import json
import time
import pandas as pd
import glob
import bert_seq2seq
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq.tokenizer import Tokenizer, load_chinese_base_vocab
from bert_seq2seq.utils import load_bert
from torch.utils.tensorboard import SummaryWriter
import torch.onnx
vocab_path = "./vocab/vocab.txt"
word2idx, keep_tokens = load_chinese_base_vocab(vocab_path=vocab_path, simplfied=True)
model_name = 'roberta'
# model_path = "./model_file/torch_model.bin"
model_path = "./model_file/robertawwm_model.bin"
model_save_path = "./model_file/trained_model/summary_roberta_wwm.bin"

batch_size = 4
lr = 1e-5
maxlen=256

df = pd.read_csv("./dataset/train_with_summ.csv")
del df["Unnamed: 0"]
train_len = df.article.__len__() // 10 * 8 
train_dataset = df[:train_len]
eval_dataset = df[train_len:]



class BertDataset(Dataset):
    """
    针对特定数据集，定义相关的取数据方式
    """
    def __init__(self):
        super(BertDataset, self).__init__()
        # 拿到数据集
        
        self.dataset = train_dataset
        # id->词
        self.idx2word = {k: v for v, k in word2idx.items()}
        # 分词器
        self.tokenizer = Tokenizer(word2idx)

    def __getitem__(self, i):
        # 得到单个数据
        summary, article = self.dataset["summary"][i], self.dataset["article"][i]
        # print(article)
        # print(summary)
        token_ids, token_type_ids = self.tokenizer.encode(
            article, summary, max_length=maxlen
        )
        output = {
            "token_ids": token_ids,
            "token_type_ids": token_type_ids,
        }
        
        return output
    
    def __len__(self):
        return len(self.dataset)

def collate_fn(batch):
    """
    动态padding，batch为一部分sample
    """
    def padding(indice, max_length, pad_idx=0):
        """
        pad函数
        """
        pad_indice = [item + [pad_idx] * max(0, max_length-len(item)) for item in indice]
        
        return torch.tensor(pad_indice)
    token_ids = [data["token_ids"]for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = (data["token_type_ids"] for data in batch)

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    # 任务目标
    target_ids_padded = token_ids_padded[:,1:].contiguous()

    return token_ids_padded, token_type_ids_padded, target_ids_padded

class Trainer:
    def __init__(self):
        # 判断是否有可用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.bert_model = load_bert(word2idx, model_name=model_name)
        ## 加载预训练的模型参数～
        
        self.bert_model.load_pretrain_params(model_path, keep_tokens=keep_tokens)
        # 加载已经训练好的模型，继续训练

        # 将模型发送到计算设备(GPU或CPU)
        self.bert_model.set_device(self.device)
        # 声明需要优化的参数
        self.optim_parameters = list(self.bert_model.parameters())
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=1e-3)
        # 声明自定义的数据加载器
        dataset = BertDataset()
        self.dataloader =  DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.tb = SummaryWriter("./tensorboard_log/summary_roberta_test_onnx")
    def train(self, epoch):
        # 一个epoch的训练
        self.bert_model.train()
        self.iteration(epoch, dataloader=self.dataloader, train=True)
    
    def save(self, save_path):
        """
        保存模型
        """
        self.bert_model.save_all_params(save_path)
        print("{} saved!".format(save_path))

    def iteration(self, epoch, dataloader, train=True):
        total_loss = 0
        start_time = time.time() ## 得到当前时间
        step = 0
        report_loss = 0
        for token_ids, token_type_ids, target_ids in tqdm(dataloader,position=0, leave=True):
            step += 1
            if step % 1000 == 0:
                self.bert_model.eval()
                test_data = eval_dataset[:200]
                for text in test_data:
                    print(self.bert_model.generate(text, beam_size=3))
                print("loss is " + str(report_loss))
                report_loss = 0
                # self.eval(epoch)
                self.bert_model.train()
            if step % 8000 == 0:
                self.save(model_save_path)

            # 因为传入了target标签，因此会计算loss并且返回
            predictions, loss = self.bert_model(token_ids,
                                                token_type_ids,
                                                labels=target_ids,
                                               
                                                )
            report_loss += loss.item()
            
            # 反向传播
            if train:
                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播, 获取新的梯度
                loss.backward()
                # 用获取的梯度更新模型参数
                self.optimizer.step()

            # 为计算当前epoch的平均loss
            total_loss += loss.item()
            self.tb.add_scalar('loss', loss, step+epoch*batch_size)
        end_time = time.time()
        spend_time = end_time - start_time
    
        # 打印训练信息
        print("epoch is " + str(epoch)+". loss is " + str(total_loss) + ". spend time is "+ str(spend_time))
        # 保存模型
        self.save(model_save_path)

if __name__ == '__main__':
    trainer = Trainer()
train_epoches = 10

for epoch in range(train_epoches):
    # 训练一个epoch
    trainer.train(epoch)