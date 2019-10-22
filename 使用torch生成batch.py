import torch

import torch.utils.data as Data


BATCH_SIZE = 5  # 批训练的数据个数，每组五个

x = torch.zeros(10,15)  # x data (torch tensor)

y = torch.ones(10,15)  # y data (torch tensor)

# 先转换成 torch 能识别的 Dataset

torch_dataset = Data.TensorDataset(x,y)

# 把 dataset 放入 DataLoader

loader = Data.DataLoader(

    dataset=torch_dataset,  # torch TensorDataset format

    batch_size=BATCH_SIZE,  # 每组的大小

    shuffle=False,  # 要不要打乱数据 (打乱比较好)

)

for epoch in range(3):  # 对整套数据训练三次，每次训练的顺序可以不同

    for step, (x, y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习

        # 假设这里就是你训练的地方...

        # 打出来一些数据

        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',

             x.numpy(), '| batch y: ', y.numpy())