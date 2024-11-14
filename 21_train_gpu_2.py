import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tudui_model import Tudui

# 定义训练的设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据集
train_data = torchvision.datasets.CIFAR10(root="dataset", train=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root="dataset", train=False, transform=torchvision.transforms.ToTensor())

train_data_size = len(train_data)
test_data_size = len(test_data)
print("train len:{}, test len:{}".format(train_data_size, test_data_size))

# DataLoader 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
tudui = Tudui()
tudui.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), learning_rate)

# 设置训练网络的参数
total_train_step = 0  # 记录训练次数
total_test_step = 0  # 记录测试次数
epoch = 10  # 训练轮数

# 添加 tensorboard
writer = SummaryWriter("logs_train")

for i in range(epoch):
    print("------第 {} 轮训练开始------".format(i + 1))

    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        # 数据，标注使用 .to(device) 需要重新赋值
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = tudui(imgs)

        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1

        if (total_train_step % 100 == 0):
            print("训练次数: {}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = tudui(imgs)

            loss = loss_fn(outputs, targets)
            total_test_loss += loss

            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("测试集上的Loss: {}".format(total_test_loss))
    print("测试集上的Accuracy: {}".format(total_accuracy / test_data_size))

    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    torch.save(tudui, "tudui_{}.pth".format(i))
    print("模型 {} 已保存".format(i + 1))

writer.close()
