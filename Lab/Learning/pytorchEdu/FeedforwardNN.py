import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 创建玩具数据集
n_samples = 100
x = torch.cat((torch.randn(n_samples, 2) - 1, torch.randn(n_samples, 2) + 1), dim=0)
y = torch.cat((torch.zeros(n_samples, 1), torch.ones(n_samples, 1)), dim=0)

# 提取每个类别的数据
x_class0 = x[y.squeeze() == 0]  # 第一类数据
x_class1 = x[y.squeeze() == 1]  # 第二类数据

# 绘制散点图
plt.scatter(x_class0[:, 0], x_class0[:, 1], color='blue', label='Class 0')
plt.scatter(x_class1[:, 0], x_class1[:, 1], color='red', label='Class 1')

# 添加图例和轴标签
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')

# 显示图像
plt.show()


# 定义前馈神经网络模型
class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedforwardNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        return x

input_size = 2
hidden_size = 10
output_size = 1
model = FeedforwardNN(input_size, hidden_size, output_size)

# 定义损失函数和优化器
loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 训练模型
num_epochs = 200
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(x)

    # 计算损失
    loss = loss_function(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印损失
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

# 创建验证集
n_samples_val = 50
x_val = torch.cat((torch.randn(n_samples_val, 2) - 1, torch.randn(n_samples_val, 2) + 1), dim=0)
y_val = torch.cat((torch.zeros(n_samples_val, 1), torch.ones(n_samples_val, 1)), dim=0)
# 提取验证集中的每个类别的数据
x_val_class0 = x_val[y_val.squeeze() == 0]
x_val_class1 = x_val[y_val.squeeze() == 1]

# 绘制验证集数据点
plt.figure()
plt.scatter(x_val_class0[:, 0], x_val_class0[:, 1], color='blue', label='Class 0')
plt.scatter(x_val_class1[:, 0], x_val_class1[:, 1], color='red', label='Class 1')

# 计算决策边界
x1_range = np.linspace(x_val[:, 0].min() - 1, x_val[:, 0].max() + 1, 100)
x2_range = np.linspace(x_val[:, 1].min() - 1, x_val[:, 1].max() + 1, 100)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
x_grid = torch.tensor(np.vstack([x1_grid.ravel(), x2_grid.ravel()]).T, dtype=torch.float32)

with torch.no_grad():
    y_grid_pred = model(x_grid)
    y_grid_pred = torch.sigmoid(y_grid_pred)
    y_grid_pred = (y_grid_pred > 0.5).float()

y_grid_pred = y_grid_pred.view(x1_grid.shape)
# 绘制决策边界
plt.contour(x1_grid, x2_grid, y_grid_pred, levels=[0.5], colors='green', linestyles='--')
plt.title('Decision Boundary')

# 添加图例和轴标签
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')

# 显示图像
plt.show()