import torch
import os, sys
sys.path.append("..")
from bert_seq2seq.tokenizer import Tokenizer, load_chinese_base_vocab
import json
import time
import bert_seq2seq
from bert_seq2seq.utils import load_bert

# 模型文件以及词表
# model = "../model_file/trained_model/summary_roberta_wwm.bin"
model = "../model_file/trained_model/summary_roberta_wwm_256.bin"
# pretrian = "./model_file/roberta_wwm.bin"
vocab_text = "../vocab/vocab.txt"
model_name = "roberta"

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    # 加载字典
    word2idx,_ = load_chinese_base_vocab(vocab_path=vocab_text,simplfied=True)
    tokenizer = Tokenizer(word2idx)
    # 定义模型
    bert_model = load_bert(word2idx,model_name=model_name)
    bert_model.set_device(device)
    bert_model.eval()
    bert_model.load_all_params(model_path=model, device=device)
    return bert_model

def predict(model, text):
    with torch.no_grad():
        res = model.generate(text, beam_size=3,out_max_length=100)
    return res