import os
import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

# ==================== 参数设置（请根据实际情况修改）====================
scale = 6                                         # k-mer 长度，和词向量训练的k-mer 保持一致
model_path = r""                                  # 你刚训练好的词向量模型
fasta_dir = r""                                   # 你的 707 个 FASTA 文件目录
output_dir = r""                                  # 输出特征文件的目录
label_file = r""                                  # 需要准备的s1_accessions_labels.txt标签文件
random_seed = 42                                  # 随机种子，保证结果可重复

os.makedirs(output_dir, exist_ok=True)

# ==================== 1. 加载词向量模型 ====================
print("正在加载词向量模型...")
model = Word2Vec.load(model_path)
vector_dim = model.wv.vector_size
print(f"词向量维度: {vector_dim}")  # 应该是 256

# ==================== 2. 读取标签表 ====================
print("正在读取标签文件...")
labels_dict = {}
with open(label_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):  # 跳过空行和注释
            continue
        parts = line.split()
        if len(parts) >= 2:
            acc = parts[0]                # Accession 号
            label_str = parts[1]          # 标签字符串
            if label_str.lower() == 'probiotics':
                labels_dict[acc] = 1
            else:
                labels_dict[acc] = 0

print(f"标签条目数: {len(labels_dict)}")

# ==================== 3. 遍历所有 FASTA 文件进行编码 ====================
features = []   # 用于存储每个基因组的特征向量
labels = []     # 用于存储对应的标签
valid_files = 0

fasta_files = [f for f in os.listdir(fasta_dir) if f.endswith('.fasta')]
print(f"找到 {len(fasta_files)} 个 FASTA 文件，开始编码...")

#for i, fasta_file in enumerate(fasta_files):
for fasta_file in tqdm(fasta_files, desc="编码基因组", unit="个"):
    # 提取 Accession 号（去掉 .fasta 后缀）
    acc = fasta_file[:-6] if fasta_file.endswith('.fasta') else fasta_file
    # 如果文件名里有其他后缀（比如 .1.fasta），上面的方法也能正确去掉 .fasta
    
    if acc not in labels_dict:
        print(f"  警告！ {acc} 没有对应的标签，跳过")
        continue

    # 读取序列（跳过以 '>' 开头的注释行）
    file_path = os.path.join(fasta_dir, fasta_file)
    seq = ''
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                continue
            seq += line.strip().upper()

    # 切 k-mer 并查词向量
    kmers = [seq[j:j+scale] for j in range(len(seq) - scale + 1)]
    vectors = [model.wv[kmer] for kmer in kmers if kmer in model.wv]

    if vectors:
        # 对所有 k-mer 的向量取平均，得到基因组的固定长度特征向量
        genome_vector = np.mean(vectors, axis=0)
    else:
        # 极少数情况：一个 k-mer 都没命中（例如序列极短），填 0 向量
        genome_vector = np.zeros(vector_dim)

    features.append(genome_vector)
    labels.append(labels_dict[acc])
    valid_files += 1

    if (valid_files) % 100 == 0:
        print(f"  已处理 {valid_files} 个基因组...")

print(f"编码完成！共成功处理 {valid_files} 个基因组。")

# ==================== 4. 转换为数组并保存为十折交叉验证格式 ====================
features = np.array(features)
labels = np.array(labels)

print(f"特征矩阵形状: {features.shape}")
print(f"标签向量形状: {labels.shape}")

# 随机打乱（固定随机种子，确保可重复）
np.random.seed(random_seed)
indices = np.random.permutation(len(features))
features = features[indices]
labels = labels[indices]

# 按 9 : 1 划分训练集和测试集（分层抽样，保持类别比例）
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.1, random_state=random_seed, stratify=labels
)

# 将训练集均匀分成 10 份，分别保存为 test_0.npy ~ test_9.npy
# 测试集保存为 test_10.npy
num_train_folds = 10
train_size = len(X_train)
fold_size = train_size // num_train_folds

for fold in range(num_train_folds):
    start = fold * fold_size
    end = start + fold_size if fold != num_train_folds - 1 else train_size
    X_fold = X_train[start:end]
    y_fold = y_train[start:end]
    np.save(os.path.join(output_dir, f"My_test_{fold}.npy"), X_fold)
    np.save(os.path.join(output_dir, f"My_test_label_{fold}.npy"), y_fold)

# 保存测试集
np.save(os.path.join(output_dir, f"My_test_{num_train_folds}.npy"), X_test)
np.save(os.path.join(output_dir, f"My_test_label_{num_train_folds}.npy"), y_test)

print(f"特征文件已保存到 {output_dir}")
print(f"训练集总样本数: {len(X_train)}，测试集样本数: {len(X_test)}")
print("请确认输出目录包含以下文件：")
print("  My_test_0.npy ... My_test_9.npy   (训练集)")
print("  My_test_label_0.npy ... My_test_label_9.npy (训练标签)")
print("  My_test_10.npy (测试集)")
print("  My_test_label_10.npy (测试标签)")