import copy
import datetime
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

# ------------------ 参数 ------------------
log_name = f"CT_RoBERTa_8_log.txt"
model_name = "CT_RoBERTa_8"
vector_fold = "embed_RoBERTa_binslong_8"
input_dim = 256

# log_name = f"CT_Word2vec_8_log.txt"
# model_name = "CT_Word2vec_8"
# vector_fold = "embed_Word2vec_binslong_8"
# input_dim = 100
# ------------------ 日志 ------------------
def write_log(content):
    with open(log_name, 'a+') as f:
        f.write(content + "\n")
        f.flush()

with open(log_name, 'w') as file:
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file.write(f"Last update time: {current_time}\n")

X_train = []
y_train = []

X_test = []
y_test = []
index = 0
for index in range(11):
    X = np.load(f"/lustrefs/home/bmze/tyh_workplace/self_Bert/vectors/{vector_fold}/test_" + str(index) + ".npy", allow_pickle=True)
    y = np.load(f"/lustrefs/home/bmze/tyh_workplace/self_Bert/vectors/{vector_fold}/test_label_" + str(index) + ".npy", allow_pickle=True)
    if index <= 9:
        for line in range(len(y)):
            X_train.append(X[line])
            y_train.append(y[line])
    elif index == 10:
        for line in range(len(y)):
            X_test.append(X[line])
            y_test.append(y[line])


write_log("数据加载完成！")

X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train)
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test)
# 数据加载器
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 定义一维CNN分类器
import torch
import torch.nn as nn

class CNNTransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes,
                 embed_dim=128, num_heads=4, num_layers=2, ffn_dim=input_dim*4, dropout=0.1):
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
        self.fc1 = nn.Linear(embed_dim, 256)
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
        # x = x.mean(dim=1)  # (batch, embed_dim)

        # 分类头
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.fc2(x)
        return x


# 实例化模型

# input_dim = 100
num_classes = 2
model = CNNTransformerClassifier(input_dim, num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.000005)

loss_set = []
acc_set = []
best_acc = 0
best_model = ""
# 训练模型
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 记录损失
    loss_value = loss.item()
    loss_set.append(loss_value)
    write_log(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_value:.4f}')
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_value:.4f}')

    # **更新学习率**
    scheduler.step()

    # **打印当前学习率**
    current_lr = scheduler.get_last_lr()[0]
    write_log(f"Epoch [{epoch + 1}], Current LR: {current_lr:.6f}")
    print(f"Epoch [{epoch + 1}], Current LR: {current_lr:.6f}")

    # 测试模型
    model.eval()
    with torch.no_grad():
        all_labels = []
        all_preds = []
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        acc_set.append(accuracy)
        if best_acc < accuracy:
            best_acc = accuracy
            best_model = copy.deepcopy(model)
        write_log(f'Test Accuracy: {accuracy:.4f}, best_acc: {best_acc}')

write_log(f"保存最好模型的精确率为：{best_acc}")
torch.save(best_model, f"./{model_name}")
