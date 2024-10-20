import time

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# prepare 數據集
train_data = torchvision.datasets.CIFAR10("../data", True, torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10("../data", False, torchvision.transforms.ToTensor(), download=True)

# length 長度
train_data_size = len(train_data)
test_data_size = len(test_data)

# 如果train
print("訓練數據集的長度為：{}".format(train_data_size))
print("測試數據集的長度為：{}".format(test_data_size))

# 利用DataLoader 來加載數據集
train_dataloader = DataLoader(train_data, 64)
test_dataloader = DataLoader(test_data, 64)

# 設置執行設備
# device = torch.device("cpu")
device = torch.device("cuda:0")

class Yukai(nn.Module):
    def __init__(self):
        super(Yukai, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 創建網路模型
yk = Yukai()
# if torch.cuda.is_available():
#     print("cuda OK!")
# yk = yk.cuda()
yk = yk.to(device)

# 損失函數
loss_fun = nn.CrossEntropyLoss()
# if torch.cuda.is_available():
#     loss_fun = loss_fun.cuda()
loss_fun = loss_fun.to(device)

# 優化器
# learning rate = 0.001
learning_rate = .001
optimizer = torch.optim.SGD(yk.parameters(), lr=learning_rate)

# 設置訓練網路的一些參數
# 記錄訓練的次數
total_train_step = 0
# 記錄測試的次數
total_test_step = 0
# 訓練的輪數
epoch = 10

writer = SummaryWriter("./logs_train")

for i in range(epoch):
    print(f"---------第{i} 輪訓練開始---------")
    start_time = time.time()

    # 訓練部驟開始
    for data in train_dataloader:
        imgs, targets = data
        # if torch.cuda.is_available():
        #     imgs = imgs.cuda()
        #     targets = targets.cuda()
        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = yk(imgs)
        loss = loss_fun(outputs, targets)

        # 優化器優化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1

        if total_train_step % 100 == 0:
            print(f"訓練次數：{total_train_step}，loss={loss}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)
            end_time = time.time()
            calculation_time = end_time - start_time
            print(f"這輪到目前為止花費了{calculation_time}s")

    # 測試步驟開始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data

            # if torch.cuda.is_available():
            #     imgs = imgs.cuda()
            #     targets = targets.cuda()
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = yk(imgs)
            loss = loss_fun(outputs, targets)
            total_test_loss = total_test_loss + loss
            accurcy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accurcy

    print(f"整體測試集上的Loss: {total_test_loss}")
    print(f"整體測試集上的正確率: {total_accuracy / test_data_size}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(yk, "yk_{}_.pth".format(i))
    print("模型已保存！")
writer.close()
