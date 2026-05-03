# coding=utf-8
import os
import re

from Bio import Entrez, SeqIO

name_list = []
with open("../data/prok_reference_genomes.txt", encoding='utf-8') as file:
    lines = file.readlines()
    for line in lines:
        line = re.split(' |,|\t', line)
        for word in line:
            if "NC_" in word:
                name_list.append(word)
print(name_list)
print(len(name_list))

n = 0
for name in name_list:
    Entrez.email = "T1425967005@outlook.com"
    hd1 = Entrez.efetch(db="nucleotide", id=[name], rettype='fasta')
    seq = SeqIO.read(hd1, 'fasta')
    fw = open("../data/bio_120/" + name + ".fasta", 'w')
    SeqIO.write(seq, fw, 'fasta')
    fw.close()
    os.getcwd()
    n = n + 1
    print("第" + str(n) + "个序列" + name + "下载完成！！")