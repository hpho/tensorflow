#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# https://github.com/golbin/TensorFlow-Tutorials/blob/master/09%20-%20GAN/01%20-%20GAN.py
# 2016년에 가장 관심을 많이 받았던 비감독(Unsupervised) 학습 방법인
# Generative Adversarial Network(GAN)을 구현해봅니다.
# https://arxiv.org/abs/1406.2661
import tensorflow as tf
import numpy as np
import pandas as pd

Data = pd.read_csv(r'D:\Desktop\SynologyDrive\하강우\bubble\data\bubble.csv')

def stz(Data):
    return (Data - Data.mean())/Data.std()

def norm(Data, min_norm=-0.9, max_norm=0.9):
    return (Data - Data.min())/(Data.max() - Data.min())*(max_norm - min_norm) + min_norm

def return_stz(stz_Data, raw_Data):
    return stz_Data*raw_Data.std() + raw_Data.mean()

def return_norm(norm_Data, raw_Data, min_norm=-0.9, max_norm=0.9):
    return (norm_Data - min_norm)/(max_norm - min_norm)*(raw_Data.max() - raw_Data.min()) + raw_Data.min()

min_norm, max_norm = -0.9 , 0.9

stz_Data = stz(Data)
norm_Data = norm(Data, min_norm, max_norm)

All_Data = pd.DataFrame()
All_Data[Data.columns[0:3]] = norm_Data[Data.columns[0:3]]
All_Data[Data.columns[3]] = stz_Data[Data.columns[3]]
All_Data[Data.columns[4]] = norm_Data[Data.columns[4]]
All_Data[Data.columns[5]] = stz_Data[Data.columns[5]]


#########
# 옵션 설정
######
total_epoch = 200
batch_size = 20
learning_rate = 0.0002
# 신경망 레이어 구성 옵션
n_hidden = 20
n_input = 6
n_noise = 6  # 생성기의 입력값으로 사용할 노이즈의 크기

#########
# 신경망 모델 구성
######
# GAN 도 Unsupervised 학습이므로 Autoencoder 처럼 Y 를 사용하지 않습니다.
X = tf.placeholder(tf.float32, [None, n_input])
# 노이즈 Z를 입력값으로 사용합니다.
Z = tf.placeholder(tf.float32, [None, n_noise])

# 생성기 신경망에 사용하는 변수들입니다.
G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev=0.01))
G_b1 = tf.Variable(tf.zeros([n_hidden]))
G_W2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01))
G_b2 = tf.Variable(tf.zeros([n_input]))

# 판별기 신경망에 사용하는 변수들입니다.
D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([n_hidden]))
# 판별기의 최종 결과값은 얼마나 진짜와 가깝냐를 판단하는 한 개의 스칼라값입니다.
D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
D_b2 = tf.Variable(tf.zeros([1]))


# 생성기(G) 신경망을 구성합니다.
def generator(noise_z):
    hidden = tf.nn.relu(
                    tf.matmul(noise_z, G_W1) + G_b1)
    output = tf.nn.sigmoid(
                    tf.matmul(hidden, G_W2) + G_b2)

    return output


# 판별기(D) 신경망을 구성합니다.
def discriminator(inputs):
    hidden = tf.nn.relu(
                    tf.matmul(inputs, D_W1) + D_b1)
    output = tf.nn.sigmoid(
                    tf.matmul(hidden, D_W2) + D_b2)

    return output


# 랜덤한 노이즈(Z)를 만듭니다.
def get_noise(batch_size, n_noise):
    return np.random.normal(size=(batch_size, n_noise))


# 노이즈를 이용해 랜덤한 이미지를 생성합니다.
G = generator(Z)
# 노이즈를 이용해 생성한 이미지가 진짜 이미지인지 판별한 값을 구합니다.
D_gene = discriminator(G)
# 진짜 이미지를 이용해 판별한 값을 구합니다.
D_real = discriminator(X)

# 논문에 따르면, GAN 모델의 최적화는 loss_G 와 loss_D 를 최대화 하는 것 입니다.
# 다만 loss_D와 loss_G는 서로 연관관계가 있기 때문에 두 개의 손실값이 항상 같이 증가하는 경향을 보이지는 않을 것 입니다.
# loss_D가 증가하려면 loss_G는 하락해야하고, loss_G가 증가하려면 loss_D는 하락해야하는 경쟁관계에 있기 때문입니다.
# 논문의 수식에 따른 다음 로직을 보면 loss_D 를 최대화하기 위해서는 D_gene 값을 최소화하게 됩니다.
# 판별기에 진짜 이미지를 넣었을 때에도 최대값을 : tf.log(D_real)
# 가짜 이미지를 넣었을 때에도 최대값을 : tf.log(1 - D_gene)
# 갖도록 학습시키기 때문입니다.
# 이것은 판별기는 생성기가 만들어낸 이미지가 가짜라고 판단하도록 판별기 신경망을 학습시킵니다.
loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))
# 반면 loss_G 를 최대화하기 위해서는 D_gene 값을 최대화하게 되는데,
# 이것은 가짜 이미지를 넣었을 때, 판별기가 최대한 실제 이미지라고 판단하도록 생성기 신경망을 학습시킵니다.
# 논문에서는 loss_D 와 같은 수식으로 최소화 하는 생성기를 찾지만,
# 결국 D_gene 값을 최대화하는 것이므로 다음과 같이 사용할 수 있습니다.
loss_G = tf.reduce_mean(tf.log(D_gene))

# loss_D 를 구할 때는 판별기 신경망에 사용되는 변수만 사용하고,
# loss_G 를 구할 때는 생성기 신경망에 사용되는 변수만 사용하여 최적화를 합니다.
D_var_list = [D_W1, D_b1, D_W2, D_b2]
G_var_list = [G_W1, G_b1, G_W2, G_b2]

# GAN 논문의 수식에 따르면 loss 를 극대화 해야하지만, minimize 하는 최적화 함수를 사용하기 때문에
# 최적화 하려는 loss_D 와 loss_G 에 음수 부호를 붙여줍니다.
train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D,
                                                         var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G,
                                                         var_list=G_var_list)

#########
# 신경망 모델 학습
######
sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(len(Data)/batch_size)
loss_val_D, loss_val_G = 0, 0

for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_x = All_Data.sample(n=batch_size)
        noise = get_noise(batch_size, n_noise)
        
        _, loss_val_D = sess.run([train_D, loss_D],feed_dict={X: batch_x, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G],feed_dict={Z: noise})
        
        print('Epoch:', '%04d' % epoch,'D loss: {:.4}'.format(loss_val_D),'G loss: {:.4}'.format(loss_val_G))
        


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

sample_size = 100
# 100개의 6차원 데이터 생성 ( Re, Jg/Jf , alpha_G, uf/Jf, ug/Jf, Dsm/R)

noise = get_noise(sample_size, n_noise)
samples = sess.run(G, feed_dict={Z:noise})
generate_Data = pd.DataFrame(samples,columns=Data.columns)

return_Data = pd.DataFrame()
return_Data[Data.columns[0:3]] = return_norm(generate_Data[Data.columns[0:3]], Data[Data.columns[0:3]], min_norm, max_norm)
return_Data[Data.columns[3]] = return_stz(generate_Data[Data.columns[3]],Data[Data.columns[3]])
return_Data[Data.columns[4]] = return_norm(generate_Data[Data.columns[4]], Data[Data.columns[4]], min_norm, max_norm)
return_Data[Data.columns[5]] = return_stz(generate_Data[Data.columns[5]],Data[Data.columns[5]])

return_Data.head()
# In[ ]:

plt.scatter(return_Data[return_Data.columns[0]],return_Data[return_Data.columns[1]],color='red',s=10,label='Generated data by GAN')
plt.scatter(Data[Data.columns[0]],Data[Data.columns[1]],color='blue',s=10,label='Experiment data')

plt.grid(True)
plt.legend()
