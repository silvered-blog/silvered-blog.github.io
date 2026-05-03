import os

# ==================== 参数设置（请根据实际情况修改）====================
predict_file = r".\predict_result_shiyong_My_CNN_3Conv.txt"  # predict.py 或者 My_predict.py 输出的结果文件路径
label_file = r".\s1_accessions_labels.txt"    # 标签文件路径
# ===================================================================

# 1. 读取真实标签，建立字典：accession -> 0/1
true_labels = {}
with open(label_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 2:
            acc = parts[0].strip()
            label_str = parts[1].strip()
            true_labels[acc] = 1 if label_str.lower() == 'probiotics' else 0

# 2. 读取预测结果，建立字典：accession -> 0/1
pred_labels = {}
with open(predict_file, 'r', encoding='utf-8') as f:
    for line in f:
        # 每行格式示例： NC_000913.fasta -> 预测类别: 1, 当前益生菌总数: 1
        if '预测类别:' not in line:
            continue
        parts = line.split('预测类别:')
        if len(parts) < 2:
            continue
        left_part = parts[0].strip()       # 文件名部分
        right_part = parts[1].strip()      # "1, 当前益生菌总数: 1"

        # 提取文件名（去掉路径，可能含 .fasta）
        filename = os.path.basename(left_part.split('->')[0].strip())
        # 去掉 .fasta 后缀得到 accession
        if filename.endswith('.fasta'):
            accession = filename[:-6]
        else:
            accession = filename

        # 提取预测数字（逗号前的部分）
        pred_str = right_part.split(',')[0].strip()
        try:
            pred = int(pred_str)
        except:
            continue
        pred_labels[accession] = pred

# 3. 计算准确率
correct = 0
total = 0
for acc, true_label in true_labels.items():
    if acc in pred_labels:
        if pred_labels[acc] == true_label:
            correct += 1
        total += 1

accuracy = correct / total if total > 0 else 0

# 4. 计算精确率、召回率、F1
# 益生菌为正类 (1)
tp = 0   # 预测为益生菌，实际也是益生菌
fp = 0   # 预测为益生菌，实际不是
fn = 0   # 预测为非益生菌，实际是益生菌
tn = 0   # 预测为非益生菌，实际也不是

for acc, true_label in true_labels.items():
    if acc in pred_labels:
        pred = pred_labels[acc]
        if true_label == 1 and pred == 1:
            tp += 1
        elif true_label == 1 and pred == 0:
            fn += 1
        elif true_label == 0 and pred == 1:
            fp += 1
        elif true_label == 0 and pred == 0:
            tn += 1

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

# 5. 输出结果
print("=" * 50)
print(f"总样本数: {total}")
print(f"正确预测数: {correct}")
print(f"准确率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
print("-" * 50)
print(f"精确率 (Precision): {precision:.4f} ({precision*100:.2f}%)")
print(f"召回率 (Recall):    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1 分数:            {f1:.4f}")
print("=" * 50)
print(f"混淆矩阵:")
print(f"              预测益生菌  预测非益生菌")
print(f"实际益生菌      {tp:6d}        {fn:6d}")
print(f"实际非益生菌    {fp:6d}        {tn:6d}")