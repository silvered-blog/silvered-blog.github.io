import os
import datetime
import time
import torch
import torch.nn as nn
import numpy as np
import argparse
import multiprocessing
from gensim.models import Word2Vec
from tqdm import tqdm
# import cnn_transformer
# import cnn
# import cnn_gru


# ========== 工具函数 ==========
def write_log(log_path, content):
    with open(log_path, 'a+', encoding='utf-8') as f:
        f.write(content + "\n")
        f.flush()


def load_embedding_txt(txt_path):
    embedding_dict = {}
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            token = parts[0]
            vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            embedding_dict[token] = vector
    return embedding_dict


def clean_dna_sequence(sequence):
    """清洗DNA序列，只保留ATCG"""
    return ''.join([b for b in sequence.upper() if b in 'ATCG'])


def seq_to_vec(seq, embedding_dict, k):
    """将DNA序列转为平均embedding向量"""
    kmers = [seq[i:i + k] for i in range(len(seq)) if seq[i:i + k] in embedding_dict]
    if kmers:
        return np.mean([embedding_dict[k] for k in kmers], axis=0)
    else:
        return np.zeros(len(next(iter(embedding_dict.values()))), dtype=np.float32)


# ========== 模型定义 ==========
class CNNTransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes,
                 embed_dim=128, num_heads=4, num_layers=2, ffn_dim=400, dropout=0.1):
        super(CNNTransformerClassifier, self).__init__()

        # 1D卷积：提取局部特征
        self.conv = nn.Conv1d(in_channels=1, out_channels=embed_dim,
                              kernel_size=(5,), padding=2)
        self.batchnorm = nn.BatchNorm1d(embed_dim)
        self.activation = nn.LeakyReLU(0.1)
        self.pool = nn.MaxPool1d(kernel_size=2)  # 降维

        # Transformer编码层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True  # 让输入为 (batch, seq_len, embed_dim)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头
        reduced_dim = input_dim // 2  # CNN池化一次后长度减半
        self.fc1 = nn.Linear(embed_dim * reduced_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 输入 x: (batch, seq_len)
        x = x.unsqueeze(1)  # (batch, 1, seq_len)

        # CNN模块
        x = self.conv(x)               # (batch, embed_dim, seq_len)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.pool(x)               # (batch, embed_dim, seq_len/2)

        # Transformer需要 (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)         # (batch, seq_len/2, embed_dim)

        # Transformer编码
        x = self.transformer(x)        # (batch, seq_len/2, embed_dim)

        # 展平
        x = x.reshape(x.size(0), -1)   # (batch, embed_dim * (seq_len/2))

        # 分类头
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.fc2(x)
        return x


class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNN1DClassifier, self).__init__()

        # 激活函数
        self.activation = nn.LeakyReLU(0.1)  # 可换成 nn.SiLU()

        # 3 层卷积
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2)

        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Dropout和BatchNorm层
        self.dropout = nn.Dropout(0.5)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(256)

        # 计算全连接层输入维度
        reduced_dim = input_dim // 2 // 2 // 2  # 经过3次池化，长度减少8倍
        self.fc1 = nn.Linear(256 * reduced_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, seq_len)

        # 3层卷积 + BatchNorm + 激活 + 池化
        x = self.activation(self.batchnorm1(self.conv1(x)))
        x = self.pool(x)

        x = self.activation(self.batchnorm2(self.conv2(x)))
        x = self.pool(x)

        x = self.activation(self.batchnorm3(self.conv3(x)))
        x = self.pool(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.fc2(x)
        return x

# ========== 多进程编码辅助函数 ==========
def encode_one_sequence(args):
    """子进程调用函数"""
    seq, embedding_dict, k = args
    return seq_to_vec(seq, embedding_dict, k)


# ========== 主流程 ==========
def run_prediction(model_path, embedding_path, input_dir, output_path, batch_size=256):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(f"启动时间: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")

    # 1️⃣ 加载模型
    write_log(output_path, f"加载模型：{model_path}")
    model = torch.load(model_path[0], map_location=device, weights_only=False)
    model.eval()

    # 2️⃣ 加载embedding
    write_log(output_path, f"加载embedding：{embedding_path}")
    embedding_dict = {}
    if model_path[4] == 'W':
        Wmodel = Word2Vec.load(embedding_path)
        embedding_dict = {word: Wmodel.wv[word] for word in Wmodel.wv.index_to_key}
    elif model_path[4] == 'R':
        embedding_dict = load_embedding_txt(embedding_path)

    # 3️⃣ 读取所有文件
    write_log(output_path, f"读取输入目录：{input_dir}")
    file_list = [os.path.join(root, f) for root, _, files in os.walk(input_dir) for f in files]
    sequences, names = [], []
    for filepath in tqdm(file_list, desc="读取序列文件", ncols=80):
        with open(filepath) as f:
            seq = clean_dna_sequence("".join(line.strip() for line in f))
        sequences.append(seq)
        names.append(os.path.basename(filepath))

    write_log(output_path, f"共读取到 {len(sequences)} 个序列文件。")

    # 4️⃣ 并行编码
    # ========== 并行编码 ==========
    cpu_cores = multiprocessing.cpu_count()
    write_log(output_path, f"开始并行序列编码... (使用CPU核心数: {cpu_cores})")

    # ✅ 新增：每完成2%写一次进度到日志
    progress_interval = max(1, len(sequences) // 50)
    results = []
    with multiprocessing.Pool(cpu_cores) as pool:
        for idx, vec in enumerate(
                tqdm(pool.imap(encode_one_sequence, [(seq, embedding_dict, model_path[1]) for seq in sequences]),
                     total=len(sequences),
                     desc="Encoding (multi-core)",
                     ncols=80)
        ):
            results.append(vec)
            if (idx + 1) % progress_interval == 0 or (idx + 1) == len(sequences):
                percent = (idx + 1) / len(sequences) * 100
                write_log(output_path, f"编码进度: {idx + 1}/{len(sequences)} ({percent:.1f}%)")

    X_vec = np.array(results, dtype=np.float32)
    write_log(output_path, "序列编码完成。")

    # 5️⃣ 推理
    write_log(output_path, "开始批量预测...")
    total = 0
    for i in tqdm(range(0, len(X_vec), batch_size), desc="预测中", ncols=80):
        batch = torch.tensor(X_vec[i:i + batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        for name, pred in zip(names[i:i + batch_size], preds):
            if pred == 1:
                total += 1
            write_log(output_path, f"{name} -> 预测类别: {pred}, 当前益生菌总数: {total}")

    # ✅ 记录时间
    elapsed = time.time() - start_time
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
    write_log(output_path, f"\n预测完成，总益生菌数: {total}")
    write_log(output_path, f"总运行时长: {elapsed_str}")
    write_log(output_path, f"结束时间: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"✅ 预测完成！总益生菌数: {total}，运行时长: {elapsed_str}")
    print(f"结果已保存到：{output_path}")


# ========== 命令行入口 ==========
if __name__ == "__main__":
    embedding_w6 = '/lustrefs/home/bmze/tyh_workplace/Mulscale_CNN/word_model/MulWord_6mer_model'
    embedding_w8 = '/lustrefs/home/bmze/tyh_workplace/Mulscale_CNN/word_model/MulWord_8mer_model'
    embedding_r8 = "/lustrefs/home/bmze/tyh_workplace/self_Bert/RoBERTa_embeddings/kmer_8mer_embeddings.txt"
    model_1 = ["./data/CT_Word2vec_6_99.58", 6, embedding_w6, 'CTW6', 'W']
    model_2 = ["./data/CT_Word2vec_8_99.06", 8, embedding_w8, 'CTW8', 'W']
    model_3 = ["./data/CNN_3Conv_RoBERTa_8_98.17", 8, embedding_r8, 'CNNR8', 'R']
    model_4 = ["./data/CNN_3Conv_Word2vec_8_99.11", 8, embedding_w8, 'CNNW8', 'W']
    model_5 = ["./data/CNNGRU_Word2vec_6_99.72", 6, embedding_w6, 'CGRUW6', 'W']
    model_6 = ["./data/CNNGRU_Word2vec_8_98.40", 8, embedding_w8, 'CGRUW8', 'W']
    model_7 = ["./data/CNN_3Conv_Word2vec_6_97.32", 6, embedding_w6, 'CNNW6', 'W']
    model_8 = ["./data/CT_RoBERTa_8_97.04", 8, embedding_r8, 'CTR8', 'R']
    model_9 = ["./data/CT_RoBERTa_8_98.08", 8, embedding_r8, 'CTR8', 'R']
    model_10 = ["./data/CT_Word2vec_8_98.59", 8, embedding_w8, 'CTW8', 'W']
    My_model = [r"D:\gitee\model-master\My_Classification_Model\CNN_3Conv_Word2vec_6", 6, r"D:\gitee\model-master\My_Word-vector_Modle\my_word2vec_6mer_model_256", 'My_CNN_3Conv', 'W']

    model = My_model
    # 使用示例
    parser = argparse.ArgumentParser(description="DNA序列益生菌预测工具")
    parser.add_argument("--input_dir", default='D:\gitee\model-master\data\genomes',
                        help="输入DNA序列文件夹")
    # parser.add_argument("--input_dir", default='/lustrefs/home/bmze/tyh_workplace/self_Bert/data/HP_selected',
    #                     help="输入DNA序列文件夹")
    parser.add_argument("--model", default=model, help="模型路径 (.pt/.pth)")
    parser.add_argument("--embedding", default=model[2], help="embedding 文件路径")
    parser.add_argument("--output", default=f"./predict_result_shiyong_{model[3]}.txt", help="输出日志文件路径")
    parser.add_argument("--batch_size", type=int, default=256, help="批量大小")
    args = parser.parse_args()

    run_prediction(
        model_path=args.model,
        embedding_path=args.embedding,
        input_dir=args.input_dir,
        output_path=args.output,
        batch_size=args.batch_size
    )
