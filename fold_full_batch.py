import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import sys
mod = sys.modules[__name__]
from functools import partial

Data = pd.read_csv(r'/home/ftmlab/Downloads/wedge_code_KW/Data/tri_KW1_wedge_Re180_Pr07.csv')

########################
del Data['X']
del Data['Y']
del Data['PHI_2-Alpha_t_lsq']
# del Data['Pr']
del Data['y_plus']
########################

# del Data['Pr']
# del Data['X']
# del Data['Y']
# del Data['alpha_t']

# del Data['S11']
del Data['S12']
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

norm_Data = norm(stz(Data))
    
train_X = norm_Data[norm_Data.columns[:-1]].sample(frac=0.7)
train_Y = norm_Data[norm_Data.columns[-1:]].iloc[train_X.index]

test_X = norm_Data[norm_Data.columns[:-1]].drop(train_X.index)
test_Y = norm_Data[norm_Data.columns[-1:]].drop(train_X.index)

num_output = 1
num_layer = 7
num_neuron = 40
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

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer(0.00001).minimize(loss)

sess = tf.Session()

for kk in range(20):

    num_overfit = 0

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    # c2_1 = []
    # c2_2 = []
    # c2_3 = []
    
    eval_index = ['train','test','total']
    for i in eval_index:
        setattr(mod,'epoch_{}'.format(i),[])
        setattr(mod,'epoch_250_{}'.format(i),[])
        setattr(mod,'filtered_epoch_1000_{}'.format(i),[])

    average_range = 3000
    
    for epoch in range(100000):

    #     batch_X = train_X.sample(n=128)
    #     batch_Y = train_Y.loc[batch_X.index]

        sess.run(train_op, feed_dict={X: train_X, Y: train_Y, training:True})       
        
        # if (epoch + 1) % 20 == 0:

        #     epoch_train.append(sess.run(loss,feed_dict={X: train_X, Y: train_Y, training:False}))
        #     epoch_test.append(sess.run(loss,feed_dict={X: test_X, Y: test_Y, training:False}))
        #     epoch_total.append(sess.run(loss,feed_dict={X: norm_Data[norm_Data.columns[:-1]], Y: norm_Data[norm_Data.columns[-1:]], training:False}))

        #     epoch_250_train.append(sess.run(loss,feed_dict={X: train_X, Y: train_Y, training:False}))
        #     epoch_250_test.append(sess.run(loss,feed_dict={X: test_X, Y: test_Y, training:False}))
        #     epoch_250_total.append(sess.run(loss,feed_dict={X: norm_Data[norm_Data.columns[:-1]], Y: norm_Data[norm_Data.columns[-1:]], training:False}))


        if (epoch+1) % average_range == 0:

            setattr(mod,'{}epoch_csv'.format(epoch+1),pd.DataFrame())
            getattr(mod,'{}epoch_csv'.format(epoch+1))['ANN'] = sess.run(model,feed_dict={X: norm_Data[norm_Data.columns[:-1]], training:False}).flatten()
                      
            for i in eval_index:
                getattr(mod,'filtered_epoch_1000_{}'.format(i)).append(np.mean(getattr(mod,'epoch_250_{}'.format(i))))
                setattr(mod,'epoch_250_{}'.format(i),[])
                
            # Criterion 1
            # if abs(filtered_epoch_1000_train[-1] - filtered_epoch_1000_test[-1])/filtered_epoch_1000_train[-1]*100 > 30:
                
            #     print('Criterion 1 is satisfied')
            #     break
                
                
            # Criterion 2
#             if len(filtered_epoch_1000_total) >= 2:

#                 if abs(filtered_epoch_1000_total[-1] - filtered_epoch_1000_total[-2])/abs(filtered_epoch_1000_total[-1]) < 0.02:
                    
#                     print('Criterion 2 is satisfied')
#                     break
#                 # c2_1.append(abs(filtered_epoch_1000_total[-1] - filtered_epoch_1000_total[-2])/abs(filtered_epoch_1000_total[-1]))                
#                 # c2_2.append(abs(filtered_epoch_1000_total[-1] - filtered_epoch_1000_total[-2])/abs(filtered_epoch_1000_total[0]))
#                 # c2_3.append(abs(filtered_epoch_1000_total[-1] - filtered_epoch_1000_total[-2])/abs(filtered_epoch_1000_total[0] - filtered_epoch_1000_total[1]))
                
# #                 if abs(filtered_epoch_500_total[-1] - filtered_epoch_500_total[-2])/abs(filtered_epoch_500_total[-1] - filtered_epoch_500_total[-2])

#             # Criterion 3
#             if filtered_epoch_1000_total[-1] < 0.0008:

#                 print('Criterion 3 is satisfied')
#                 break

    # cc = pd.DataFrame()
    # cc['train'] = epoch_train
    # cc['test'] = epoch_test
    # cc['total'] = epoch_total
    # cc.to_csv(r'/home/ftmlab/Downloads/wedge_code_KW/Save/epoch_history{}.csv'.format(kk+1),header=True,index=False)

    # dd = pd.DataFrame()
    # dd['c1'] = c2_1
    # dd['c2'] = c2_2
    # dd['c3'] = c2_3
    # dd.to_csv(r'/home/ftmlab/Downloads/wedge_code_KW/Save/criterion2_history{}.csv'.format(kk+1),header=True,index=False)

    for ss in range(int((epoch+1)/average_range)):

        getattr(mod,'{}epoch_csv'.format((ss+1)*average_range)).to_csv(r'/home/ftmlab/Downloads/wedge_code_KW/Save/S12_KW1_{}L_{}N_ANN_pred_{}_epoch_{}.csv'.format(num_layer,num_neuron,(ss+1)*average_range,kk+1),header=True,index=False)

    