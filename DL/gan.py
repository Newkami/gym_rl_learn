import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),  # 0-1归一化
    transforms.Normalize(0.5, 0.5)
])

train_ds = torchvision.datasets.MNIST('data', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)


# 生成器的输入是噪声（正态分布随机数）输出要和样本相同

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        img = self.main(x)
        img = img.view(-1, 28, 28, 1)
        return img


# 判别器的输入是生成器的样本和真实样本 输出是二分类的概率值，输出使用sigmoid
# BCELoss计算交叉熵损失
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(),  # 在负值部分保留一定的梯度
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        prob = self.main(x)
        return prob


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
gen = Generator().to(device)
dis = Discriminator().to(device)

d_optimizer = optim.Adam(dis.parameters(), lr=0.0001)
g_optimizer = optim.Adam(gen.parameters(), lr=0.0001)

loss_func = torch.nn.BCELoss()


def gen_img_plot(model, test_input):
    prediction = np.squeeze(model(test_input).detach().cpu().numpy())
    fig = plt.figure(figsize=(4, 4))
    for i in range(prediction.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((prediction[i] + 1) / 2)
        plt.axis('off')
    plt.show()


test_input = torch.randn(16, 100, device=device)

D_loss = []
G_loss = []

for epoch in range(20):
    d_epoch_loss = 0
    g_epoch_loss = 0
    count = len(dataloader)
    for step, (img, _) in enumerate(dataloader):
        img = img.to(device)
        size = img.size(0)
        random_noise = torch.randn(size, 100, device=device)
        d_optimizer.zero_grad()
        real_output = dis(img)  # 对判别器输入真实数据
        d_real_loss = loss_func(real_output, torch.ones_like(real_output))  # 判别器在真实图像上的损失
        d_real_loss.backward()

        gen_img = gen(random_noise)
        fake_output = dis(gen_img.detach())  # 优化目标为判别器 所以对生成器梯度做截断
        d_fake_loss = loss_func(fake_output, torch.zeros_like(fake_output))
        d_fake_loss.backward()
        d_loss = d_real_loss + d_fake_loss
        d_optimizer.step()

        g_optimizer.zero_grad()
        fake_output = dis(gen_img)  # 不进行梯度截断
        g_loss = loss_func(fake_output, torch.ones_like(fake_output))
        g_loss.backward()
        g_optimizer.step()

        with torch.no_grad():
            d_epoch_loss += d_loss
            g_epoch_loss += g_loss
    with torch.no_grad():
        d_epoch_loss /= count
        g_epoch_loss /= count
        D_loss.append(d_epoch_loss)
        G_loss.append(g_epoch_loss)
        print('Epoch--', epoch)
        gen_img_plot(gen, test_input)
