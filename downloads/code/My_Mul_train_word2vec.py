import datetime
from Bio.Seq import Seq
from gensim.models import word2vec
import os

scale = 6           #设置k-mer的k值
def write_log(content):
    with open(f"./Word2vec_{scale}.txt", 'a+') as f:
        f.write(content + "\n")
        f.flush()

def clean_dna_sequence(sequence):
    """去除 DNA 序列中非 ATCG 字符，并将所有字符转换为大写."""
    return ''.join([base.upper() for base in sequence if base.upper() in 'ATCG'])

def generate_complementary_sequence(dna_sequence):
    # 定义互补碱基对
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    # 生成互补序列
    complementary_sequence = ""
    for base in dna_sequence:
        if base not in ['A', 'T', 'C', 'G']:
            complementary_sequence = complementary_sequence + base
            continue
        complementary_sequence = complementary_sequence + complement[base]
    return complementary_sequence[::-1]

# 改动一：生成器函数代替列表
class MySentences(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_list = []
        for root, dirs, files in os.walk(file_path):
            if not dirs:
                self.file_list = [os.path.join(root, f) for f in files]

    def __iter__(self):
        n = 0
        for file in self.file_list:
            line_data = ''
            with open(file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if ">" in line:
                        continue
                    line = line.rstrip('\n')
                    line_data = line_data + line
            # 清洗序列
            clean_seq = clean_dna_sequence(line_data)
            # 生成正向序列的k-mer列表
            yield [clean_seq[i: i + scale] for i in range(len(clean_seq) - scale + 1)]
            # 生成反向互补序列的k-mer列表（数据增强）
            com_seq = generate_complementary_sequence(clean_seq)
            if len(com_seq) >= scale:
                yield [com_seq[i: i + scale] for i in range(len(com_seq) - scale + 1)]
            
            n += 1
            write_log("第" + str(n) + "个   " + os.path.basename(file) + "录入完成！")
            print("第" + str(n) + "个   " + os.path.basename(file) + "录入完成！")

            # 改动二：主动释放被占用的内存
            import gc
            gc.collect()

# 使用生成器，内存里只存当前处理的一个基因组
sentences = MySentences('')             #输入你的.fasta文件保存的文件夹路径

write_log(f"{scale}尺度分割完成开始训练")
print("分割完成！")

model = word2vec.Word2Vec(
    sentences,
    vector_size=256,
    hs=1,
    min_count=1,
    window=10,
    sg=1,
    epochs=10
)

model.save(f"./word2vec_{scale}mer_model_256")
