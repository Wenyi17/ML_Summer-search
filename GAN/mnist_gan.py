from keras.datasets import mnist

#我们将在MNIST数据集上训练GAN

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import sys
import numpy as np
'''
Dense — 指密集层，即上一层的每个神经元都连接到下一个输入层的每个神经元的层
Reshape — 帮助我们将图层的输出整形为特定形状。
Flatten — 允许我们通过删除除1之外的所有维来展平输入。本质上，这是将Matrix转换为简单数组的操作
Dropout —是一项重要技术，可减少过拟合模型的风险。它通过将隐藏神经元的输出边缘随机设置为0来实现。
BatchNormalization — 此机制将使我们能够训练更稳定的输入分布。这是通过将我们的输入标准化为平均值为0和标准偏差为1来实现的。
LeakyRelu —指我们的激活功能，该功能实质上将任何输入信号转换为下一层的输出信号。
Sequential — 指我们将要构建的模型的类型。这将使我们能够逐层构建模型。
Adam — 这是我们的优化器功能。
'''
class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
    optimizer = Adam(0.0002, 0.5)
    '''
    我们还指定图像的输入形状，通道和尺寸。如前所述，我们将使用Adam作为我们的优化器函数，因此在这里，我们仅将其分配给名为优化器的参数。
    '''
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
            '''
            在这里，我们构建并编译鉴别器，设置其损失函数，我们要使用的优化器以及要衡量的精度类型
            '''
        self.generator = self.build_generator()

        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        '''
        这将生成Generator并定义输入噪声。在GAN中，生成器网络将噪声z作为输入以生成其图像。
        '''
        self.discriminator.trainable = False

        validity = self.discriminator(img)
        '''
        这指定我们的鉴别器将拍摄由我们的生成器和真实数据集生成的图像，并将其输出设置为称为有效性的参数，该参数将指示输入是否真实。
        '''
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        '''
        在这里，我们组合了模型，还设置了损失函数和优化器。这里的最终目标是让生成器欺骗鉴别器。
        '''
    
    def build_generator(self):# 玩家一
    model = Sequential()
    model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))
    model.summary()
    noise = Input(shape=(self.latent_dim,))
        img = model(noise)
    return Model(noise, img)
    '''
    Alpha — α是一个超参数，它控制函数使负网络输入饱和的基础值。
    Momentum — 这是神经网络中用于加快训练速度并提高准确性的一种技术。
    '''
    def build_discriminator(self):#玩家二
    model = Sequential()
    model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
    img = Input(shape=self.img_shape)
        validity = model(img)
    return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):
        (X_train, _), (_, _) = mnist.load_data()
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
    '''
    为此，我们定义了训练函数，加载了数据集，重新缩放了训练图像并设置了基本事实。
    '''

    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        '''
        在同一循环中，我们通过设置输入噪声来训练我们的生成器，并最终通过指定梯度损失来训练生成器以使鉴别器将其样本标记为有效。
        '''
        d_loss_real = self.discriminator.train_on_batch(imgs, valid)
        d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)    
    '''
    然后，我们循环遍历多个纪元来训练鉴别器，方法是首先从真实数据集中选择一组随机图像，然后从Generator中生成一组图像，将这两组图像都输入到鉴别器中，最后为真实和伪造图像，以及综合损失。
    '''
    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5 fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()
if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=100000, batch_size=132, sample_interval=10000)
    '''
    时期决定了向前和向后传播的数量，batch_size指示了每个向后/向前传播的训练样本的数量，sample_interval指定了在多少个时期之后我们称之为sample_image函数。
    '''
