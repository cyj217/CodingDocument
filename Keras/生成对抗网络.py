import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Reshape, Flatten, Dropout, LeakyReLU, Conv2D, Conv2DTranspose, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.datasets import mnist
from tqdm import tqdm


def create_generator(latent_di, output_channels):
# 创建生成模型
generator = Sequential(name = 'generator')
# 映射并转换维度
generator.add(Dense(5*5*512, input_shape = (latent_dim,)))
generator.add(Reshape((5, 5, 512)))
generator.add(BatchNormalization())
generator.add(LeakyReLU())
# 9*9-->18*18
generator.add(Deconv2D(128, 4, strides = 2, padding = 'same'))
generator.add(BatchNormalization())
generator.add(LeakyReLU())
# 18*18-->36*36
# 采用tanh函数作为输出层激活函数，输出范围为[-1,1]
generator.add(Deconv2D(output_channels, 4, strides = 2, padding = 'same', activation = 'tanh'))
generator.summary()
return generator

def create_discriminator(input_shape):
# 创建判别模型
discriminator = Sequential(name = 'discriminator')
# 下采样到18*18
discriminator.add(Conv2D(64, 5, strides = 2, input_shape = input_shape, padding = 'same'))
discriminator.add(LeakyReLU())
# 下采样到9*9
discriminator.add(Conv2D(32, 5, strides = 2, padding = 'same'))
discriminator.add(LeakyReLU())
# 分类层
discriminator.add(Flatten())
discriminator.add(Dense(1, activation = 'sigmoid'))
discriminator.summary()
return discriminator

def train(batch_size, epochs = 100):
# 使用Adma优化器训练生成模型和判别模型
D_optimizer = Adam(1e-4, beta_1 = 0.5)
G_optimizer = Adam(1e-4, beta_1 = 0.5, decay = 1e-6)
# 构建生成模型和判别模型并只编译判别模型
G = create_generator(latent_dim, 1)
D = create_discriminator(input_shape)
D.compile(loss = 'binary_crossentropy', optimizer = D_optimizer)
# 构建生成对抗网络，保持判别模型权重不变，编译生成模型和对抗网络
D.trainable = False
GAN = Sequential([G, D], name = 'GAN')
GAN.compile(loss = 'binary_crossentropy', optimizer = G_optimizer)
# 绘制模型结构
plot_model(G, G.name + '.png')
plot_model(D, D.name + '.png')
plot_model(GAN, GAN.name + '.png')
# 构建训练数据标签，生成模型标签全为0，判别模型标签一半为0一半为1
combined_labels = np.zeros((batch_size * 2, 1))
combined_labels[:batch_size:] = 1
combined_labels = np.zeros((batch_size,1))
# 主程序训练过程
total_batchs = len(x_train) // batch_size
for e in range(epochs):
# 记录损失变换
d_losses, g_losses = [], []
for batch in range(total_batchs):
# 随机生成噪声数据作为生成模型输入
latent_noise = np.random.normal(-1, 1, size = (batch_size, latent_dim))

