import torch
import torch.nn as nn
import torch.optim as optim

# 创建数据集
x = torch.randn(20, 1)  # 20 个输入样本，每个样本有 1 个特征
y = 2 * x + 3  # 真实的线性关系，斜率为 2，截距为 3

# 定义线性回归模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)# 初始化线性层 Ax+b, A,b都是随机初始化的

    def forward(self, x):
        return self.linear(x)

model = LinearRegression()

# 定义损失函数和优化器
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 学习率为 0.01

# 训练模型
num_epochs = 100  # 迭代次数
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(x)

    # 计算损失
    loss = loss_function(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清零之前计算的梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数

    # 打印损失
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

# 打印训练后的参数
print("Trained parameters:")
print("Weight:", model.linear.weight.item())
print("Bias:", model.linear.bias.item())