import datetime
from Bio.Seq import Seq
from gensim.models import word2vec
import os

scale = 6           # k-mer要取的k值
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

def find_orfs(seq, min_len=100):
    orfs = ""
    seq = Seq(seq)
    for strand, nuc in [(+1, seq), (-1, seq.reverse_complement())]:
        for frame in range(3):
            length = 3 * ((len(nuc) - frame) // 3)  # 保证是3的倍数
            for pro in nuc[frame:frame + length].translate(table=11).split("*"):
                if len(pro) >= min_len:
                    orfs += pro
    return orfs

with open(f"./Word2vec_{scale}.txt", 'w') as file:
    # 2. 写入当前时间（格式：年-月-日 时:分:秒）
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file.write(f"最后更新时间：{current_time}")

train_data240 = []
file_path = ''                  # 输入下载的.fasta文件保存的文件夹路径
file_list = []
file_dict = {}
for iroot, idirs, ifiles in os.walk(file_path):
    if not idirs:
        file_list.extend(ifiles)

n = 0
for name in file_list:
    file = "".join([file_path, '/', name])
    line_data = ''
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if ">" in line:
                continue
            line = line.rstrip('\n')
            line_data = line_data + line
    train_data240.append(clean_dna_sequence(line_data))
    n = n + 1
    com_seq_r = generate_complementary_sequence(clean_dna_sequence(line_data))
    if len(com_seq_r) < 10:
        write_log("出错了")
        print()
    train_data240.append(com_seq_r)
    write_log("第" + str(n) + "个   " + name + "录入完成！")
    print("第" + str(n) + "个   " + name + "录入完成！")




dna_cut_list = []
for line in train_data240:
    line_cut = [line[i: i + scale] for i in range(len(line) - scale + 1)]
    dna_cut_list.append(line_cut)
write_log(f"{scale} 尺度分割完成开始训练")
print("分割完成！")
model = word2vec.Word2Vec(
    dna_cut_list,
    vector_size=256,   # 设置词向量维度
    hs=1,
    min_count=1,
    window=10,
    sg=1,
    epochs=10
)

model.save(f"./Word2vec_{scale}mer_model_256")
