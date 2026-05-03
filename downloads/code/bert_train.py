import os
import datetime

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig, AdamW
import torch.nn as nn


def write_log(content):
    with open("./Bert_train.txt", 'a+') as f:
        f.write(content + "\n")
        f.flush()


def clean_dna_sequence(sequence):
    """去除 DNA 序列中非 ATCG 字符."""
    return ''.join([base for base in sequence if base in 'ATCG'])


def reverse_complement(sequence):
    # 定义碱基互补配对字典（包括大小写）
    complement_dict = {
        'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C',
        'a': 't', 't': 'a', 'c': 'g', 'g': 'c'
    }
    reversed_complement = []
    for base in reversed(sequence):
        reversed_complement.append(complement_dict.get(base, base))
    return ''.join(reversed_complement)


with open("./Bert_train.txt", 'w') as file:
    # 2. 写入当前时间（格式：年-月-日 时:分:秒）
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file.write(f"最后更新时间：{current_time}")

# ---- Tokenizer & 模型 ----
model_name = "/lustrefs/home/bmze/tyh_workplace/Bert_project/model/DNA_bert_6"
write_log("tokenizer开始加载")
tokenizer = BertTokenizer.from_pretrained(model_name, use_fast=True)
write_log("model开始加载")
model = BertModel.from_pretrained(model_name)
write_log("开始处理数据")
# 参数
k = 6
batch_size = 8
cache_file = "cached_kmer_tokenized.pt"
max_len = 512

# 数据  DNA序列
raw_sequences = []


bio_path = "/lustrefs/home/bmze/tyh_workplace/word2vec/bio_120"


def stream_dna_sequences(file_dir):
    for name in os.listdir(file_dir):
        file_path = os.path.join(file_dir, name)
        line_data = ''
        with open(file_path, 'r') as f:
            for line in f:
                if ">" in line:
                    continue
                line_data += line.strip()
        clean_seq = clean_dna_sequence(line_data)
        yield clean_seq
        yield reverse_complement(clean_seq[::-1])  # yield 双条

def seq2kmer(seq, k=6):
    return " ".join([seq[i:i + k] for i in range(len(seq) - k + 1)])

# 初始化模型和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
model.train()

write_log("开始流式训练")

k = 6
max_len = 512
step = 0
log_interval = 100

for epoch in range(3):
    total_loss = 0
    n = 0
    for seq in stream_dna_sequences(bio_path):
        n += 1
        write_log(f"epoch {epoch} No.{n} training")
        kmer_seq = seq2kmer(seq, k)
        encoding = tokenizer.encode_plus(
            kmer_seq,
            padding='max_length',
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        dummy_target = torch.zeros_like(last_hidden)
        loss = nn.MSELoss()(last_hidden, dummy_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        step += 1

        if step % log_interval == 0:
            log_msg = f"Epoch {epoch + 1}, Step {step}, Loss: {total_loss / log_interval:.4f}"
            write_log(log_msg)
            print(log_msg)
            total_loss = 0

# 保存 embedding_matrix
embedding_layer = model.module.embeddings if isinstance(model, nn.DataParallel) else model.embeddings
embedding_matrix = embedding_layer.word_embeddings.weight.detach().cpu().numpy()
# 保存 embedding_matrix
np.save("bert_finetuned_embedding.npy", embedding_matrix)


# 保存模型（兼容单/多 GPU）
if isinstance(model, nn.DataParallel):
    torch.save(model.module.state_dict(), "bert_finetuned_model.pt")
else:
    torch.save(model.state_dict(), "bert_finetuned_model.pt")
