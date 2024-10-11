import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        # 两个全连接层
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层
        self.relu = nn.ReLU()                         # ReLU激活函数
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, num_classes) # 隐藏层到输出层

    def forward(self, x):
        out = self.fc1(x)      # 通过第一层
        out = self.relu(out)   # ReLU激活
        out = self.fc2(out)    # 通过第三层
        out = self.relu(out)   # ReLU激活
        out = self.fc3(out)    # 通过输出层
        out = self.relu(out)   # ReLU激活
        out = self.fc4(out)    # 通过输出层
        out = self.relu(out)   # ReLU激活
        out = self.fc5(out)
        return out

