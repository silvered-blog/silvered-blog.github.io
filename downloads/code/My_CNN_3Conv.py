import copy
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

# ------------------ 参数 ------------------
log_name = f"CNN_3Conv_Word2vec_6_log.txt"
model_name = "CNN_3Conv_Word2vec_6"
vector_fold = "embed_Word2vec_binslong_6"
input_dim = 256
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
    X = np.load(f"" + str(index) + ".npy", allow_pickle=True)       # 数据集文件路径
    print(f"文件 test_{index}.npy 的形状是: {X.shape}")
    y = np.load(f"" + str(index) + ".npy", allow_pickle=True)       # 标签数据集文件路径
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


# 实例化模型

# input_dim = 100
num_classes = 2
model = CNN1DClassifier(input_dim, num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.000005)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

loss_set = []
acc_set = []
best_acc = 0
best_model = ""
# path = "/lustrefs/home/bmze/tyh_workplace/self_Bert/"
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
