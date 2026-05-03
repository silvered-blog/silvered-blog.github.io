# coding=utf-8
import os
import re
import time

from Bio import Entrez, SeqIO

name_list = []
with open(r"D:\gitee\model-master\data\s1_accessions.txt", encoding='utf-8') as file:
    lines = file.readlines()
    for line in lines:
        #line = re.split(' |,|\t', line)
        acc = line.strip()
        if acc:
            name_list.append(acc)
        #for word in line:
            #if ("NC_" in word) or ("NZ_" in word):
                #name_list.append(word)
print(name_list)
print(len(name_list))

n = 0
Entrez.email = ""       #你自己的邮箱 
Entrez.api_key = ""     #你的NCBI密钥

output_dir = r""        #你要将.fasta文件保存在哪个文件夹下，输入文件夹路径
n = 0
skipped = 0

for name in name_list:

    filepath = os.path.join(output_dir, name + ".fasta")
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        print(f"⏭ {name} 已存在，跳过")
        skipped += 1
        print("已跳过：", skipped, "个")
        continue  # 直接进入下一个序列

    hd1 = Entrez.efetch(db="nucleotide", id=[name], rettype='fasta')
    seq = SeqIO.read(hd1, 'fasta')
    fw = open(output_dir + name + ".fasta", 'w')
    SeqIO.write(seq, fw, 'fasta')
    fw.close()
    os.getcwd()
    n = n + 1
    print("第" + str(skipped + n) + "个序列" + name + "下载完成！！", flush = True)
    time.sleep(0.1)
    