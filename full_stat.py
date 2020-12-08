import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import sys
mod = sys.modules[__name__]
from functools import partial

Data = pd.read_csv(r'/home/ftmlab/Downloads/wedge_code_KW/Data/trimmed_new_case2_Re180_Pr07.csv')

# #######################
# del Data['X']
# del Data['Y']
# del Data['PHI_2-Alpha_t_lsq']
# # del Data['Pr']
# del Data['y_plus']
# #######################

# del Data['Pr']
# del Data['X']
# del Data['Y']
# del Data['alpha_t']

# del Data['S11']
# del Data['S12']
# del Data['S22']
# # del Data['Sij']

# del Data['dT/dx']
# del Data['dT/dy']
# # del Data['nu_t']

# del Data['Mean_tke_pre_diff']
# del Data['Mean_tke_diss']
# del Data['Mean_tke_mol_diff']
# del Data['Mean_tfe_diss']
# del Data['Mean_tfe_mol_diff']

# # del Data['dp/dx']
# # del Data['dp/dy']

print(Data.head())

def stz(Data):
    return (Data - Data.mean())/Data.std()

def norm(Data,norm_min=0, norm_max=1):
    return (Data - Data.min())*(norm_max - norm_min)/(Data.max()- Data.min()) + norm_min

def stat_loss(rmse,grad):
    """
    rmse, grad is list type
    """
    return np.mean(rmse),np.std(rmse),np.mean(grad),np.std(grad)


norm_Data = norm(stz(Data))

num_output = 1
num_layer = 7
num_neuron = 60
activation_function = 'elu'
# initializer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1)
# initializer = tf.keras.initializers.truncated_normal()
initializer = tf.keras.initializers.he_normal()

dense_layer = partial(tf.layers.dense, activation=None, kernel_initializer=initializer, use_bias=True, bias_initializer=initializer)

training = tf.placeholder_with_default(False, shape=[], name='training')

batch_norm_layer = partial(tf.layers.batch_normalization, training=training, momentum=0.9)

X = tf.placeholder(tf.float32, shape=[None, len(norm_Data.columns)-1])
Y = tf.placeholder(tf.float32, shape=[None, num_output])

for i in range(num_layer):
    if i == 0:
        setattr(mod, 'hidden{}'.format(i+1),dense_layer(X,num_neuron))

    else:
        setattr(mod, 'hidden{}'.format(i+1),dense_layer(getattr(mod, 'hidden{}_act'.format(i)),num_neuron))

    setattr(mod,'batch_norm{}'.format(i+1),batch_norm_layer(getattr(mod,'hidden{}'.format(i+1))))
    setattr(mod,'hidden{}_act'.format(i+1),getattr(tf.nn,activation_function)(getattr(mod,'batch_norm{}'.format(i+1))))

model = dense_layer(getattr(mod,'hidden{}'.format(num_layer)), num_output)

loss = tf.math.sqrt(tf.reduce_mean(tf.square(model - Y)))
G_loss = tf.math.abs(tf.reduce_sum(tf.gradients(model,X)))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer(0.00001).minimize(loss)

sess = tf.Session()

rmse = []
grad = []

for kk in range(10):

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    for epoch in range(50000):

        sess.run(train_op, feed_dict={X: norm_Data[norm_Data.columns[:-1]], Y: norm_Data[norm_Data.columns[-1:]], training:True})
        
        if (epoch+1)%1000 == 0:
            print('epoch : ',epoch+1,'loss : ',sess.run(loss, feed_dict={X: norm_Data[norm_Data.columns[:-1]], Y: norm_Data[norm_Data.columns[-1:]], training:False}))
        
    rmse.append(sess.run(loss, feed_dict={X: norm_Data[norm_Data.columns[:-1]], Y: norm_Data[norm_Data.columns[-1:]], training:False}))
    grad.append(sess.run(G_loss, feed_dict={X: norm_Data[norm_Data.columns[:-1]], training:False}))
    
full_stat = stat_loss(rmse,grad)

print(full_stat)

    