import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import numpy as np
import sys
import time
mod = sys.modules[__name__]
start_time = time.time()
import pickle
from functools import partial
from multiprocessing import Process, Manager

def Work_manager(work_list, remain_parameter, num_process, start_time, limit_time, Data, selected_parameter, num_layer, num_neuron, act_f, initial, num_epochs, learning_rate, batch_size, num_Fold, regularization, batch_norm, patience_step, average_times, num_output, device, print_step):
    """
    work_list : dict type
    remain_parameter : set type
    num_process : int type 프로세스 개수
    start_time : 시간을 측정하기 위한 변수
    limit_time : 제한 시간
    나머지 인자들  ~ 함수 FCNN에 전달하기 위한 인자들

    """
    procs = []

    for i,j in enumerate(remain_parameter):

        if time.time() - start_time > limit_time:
            break

        if work_list[j] == 0:
            p = Process(target=FCNN, args=(Data, selected_parameter, j, num_layer, num_neuron, act_f, initial, num_epochs, learning_rate, batch_size, num_Fold, regularization, batch_norm, patience_step, average_times, num_output, device[i], print_step, work_list))
            procs.append(p)
            p.start()
            
        if len(procs) % num_process == 0:
            for p in procs:
                p.join()

    for p in procs:
        p.join()

    return

def Work_manager_removal(Data, full_parameter, num_layer, num_neuron, act_f, initial, num_epochs, learning_rate, num_Fold, batch_norm, average_times, num_output, device, print_step, avg_rmse,std_rmse,avg_grad,std_grad,num_process):
    
    procs = []

    for i,r in enumerate(full_parameter):

        if avg_rmse[r] == 0:
            p = Process(target=FCNN_removal, args=(Data, full_parameter, r, num_layer, num_neuron, act_f, initial, num_epochs, learning_rate, num_Fold, batch_norm, average_times, num_output, device[i], print_step, avg_rmse,std_rmse,avg_grad,std_grad))
            procs.append(p)
            p.start()
            
        if len(procs) % num_process == 0:
            for p in procs:
                p.join()

    for p in procs:
        p.join()

    return

def FCNN(Data, selected_parameter, i, num_layer, num_neuron, act_f, initial, num_epochs, learning_rate, batch_size, num_Fold, regularization, batch_norm, patience_step, average_times, num_output, device, print_step, work_list):
    """
    Data : pandas datafrmae type 전처리된 데이터 X와 Y Fold 어떤것도 나뉘지 않은 상태
    selected_parameter : 현재까지 선택되어진 파라미터 집합 Set type으로 주어짐
    i : str type 추가적으로 선택을 고려하는 인자 후보 집합 remain_parameter 에서 하나씩 받아옴
    num_layer : int type number of layer
    num_neuron : int type number of neuron
    act_f : str type activation function ex) 'relu','elu','tanh', 'sigmoid', ... unbounded한 활성화 함수를 사용하는 것을 추천
    init : str type 'he_normal','xavier_uniform','random_uniform'
    num_epochs : int type
    learning_rate : float type 0.001 , 0.01, 0.1, ....
    batch_size : int type 학습시 한번에 사용할 batch size 크기
    num_Fold : int type 만약 0이라면 fold를 사용하지 않음
    regularization : str type 'L1' ,'L2', None
    batch_norm : bool type if True train with batch normalization
    patience_step : int type validation data 에대한 평가 값이 증가하는 step을 얼마나 용인할것 이냐?
    average_times : int type 몇번 평균 낼 것이냐?
    num_output : int type number of output parameter
    device : str type to assign gpu card '/gpu:0'
    print_step : int type print training per step
    work_list : multiprocessing.managers.DictProxy 쉽게 말하면 프로세스간 공유 인자 dictionary type 
    """
    ##### Data Folding #####################
    print('############################ {} parameter training start #############################'.format(i))
    selected_parameter.add(i)
    parameters = list(selected_parameter)
    parameters.append(Data.columns[-1])
    
    selected_Data = Data[parameters]
    shuffle_Data = selected_Data.sample(frac=1).reset_index(drop=True)
    
    try :
        Fold_length = int(len(Data)/num_Fold)
    except ZeroDivisionError :
        pass

    for j in range(num_Fold):

        if (j+1) == num_Fold:
            setattr(mod,'Fold_{}'.format(j+1),shuffle_Data.index[(j)*Fold_length:])

        else:
            setattr(mod,'Fold_{}'.format(j+1),shuffle_Data.index[(j)*Fold_length:(j+1)*Fold_length])

    ##### Network define ####################
    with tf.device(device):

        if regularization == None:
            dense_layer = partial(tf.layers.dense, activation=None,kernel_initializer=getattr(tf.keras.initializers,initial)(),use_bias=True)
        else:
            reg = getattr(tf.contrib.layers,'{}_regularizer'.format(regularization.lower()))(scale=0.005)
            dense_layer = partial(tf.layers.dense, activation=None,kernel_initializer=getattr(tf.keras.initializers,initial)(),use_bias=True,kernel_regularizer=reg)

        training = tf.placeholder_with_default(False, shape=[], name="training")
        batch_norm_layer = partial(tf.layers.batch_normalization, training=training, momentum=0.9)

        X = tf.placeholder(tf.float32,shape=(None,len(shuffle_Data.columns)-num_output), name='X')
        Y = tf.placeholder(tf.float32,shape=(None,num_output), name='Y')

        for j in range(num_layer):

            if j == 0:
                setattr(mod, 'hidden{}'.format(j+1),dense_layer(X,num_neuron))
            else:
                setattr(mod, 'hidden{}'.format(j+1),dense_layer(getattr(mod, 'hidden{}_act'.format(j)),num_neuron))

            if batch_norm == True:
                setattr(mod,'batch_norm_hidden{}'.format(j+1),batch_norm_layer(getattr(mod,'hidden{}'.format(j+1))))
                setattr(mod,'hidden{}_act'.format(j+1),getattr(tf.nn,act_f)(getattr(mod,'batch_norm_hidden{}'.format(j+1))))
            else:
                setattr(mod,'hidden{}_act'.format(j+1),getattr(tf.nn,act_f)(getattr(mod,'hidden{}'.format(j+1))))
        
        model = dense_layer(getattr(mod,'hidden{}'.format(num_layer)), num_output)

        mse_loss = tf.reduce_mean(tf.square(model - Y))

        if regularization == False:
            loss = mse_loss
        else:
            loss_regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss = tf.add_n([mse_loss] + loss_regularization)

        if batch_norm == False:
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(mse_loss)
        else:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

    ###### training start ##########################

    with tf.Session(config=config) as sess:

        test_loss_list = []

        for j in range(average_times):

            for k in range(num_Fold):
                sess.run(tf.global_variables_initializer())

                Test_X = shuffle_Data[shuffle_Data.columns[:-num_output]].iloc[getattr(mod,'Fold_{}'.format(k+1))]
                Test_Y = shuffle_Data[shuffle_Data.columns[-num_output:]].iloc[getattr(mod,'Fold_{}'.format(k+1))]
                Train_X = shuffle_Data[shuffle_Data.columns[:-num_output]].drop(Test_X.index)
                Train_Y = shuffle_Data[shuffle_Data.columns[-num_output:]].drop(Test_Y.index)

                num_overfit = 0
                test_loss = 1000
                for epoch in range(num_epochs):
                    
                    # batch = shuffle_Data.drop(range(getattr(mod,'Fold_{}'.format(k+1))[0],getattr(mod,'Fold_{}'.format(k+1))[-1]+1)).sample(n=batch_size)

                    sess.run(train_op, feed_dict={X: Train_X, Y: Train_Y, training: True})
                    
                    if test_loss <= sess.run(loss, feed_dict={X: Test_X, Y: Test_Y, training: False}):
                        num_overfit += 1
                    
                    else:
                        test_loss = sess.run(loss, feed_dict={X: Test_X, Y: Test_Y, training: False})
                        num_overfit = 0
                    
                    if num_overfit == patience_step:
                        print(epoch+1,' epochs test mse loss : ',test_loss)
                        break

                    if (epoch+1) % print_step == 0:
                        print('{} of {} training {} of {} Fold {} epoch '.format(j+1,average_times,k+1,num_Fold,epoch+1),
                              'train loss : ',sess.run(loss, feed_dict={X: Train_X, Y: Train_Y, training: False}),
                              'test loss : ',sess.run(loss, feed_dict={X: Test_X, Y: Test_Y, training: False}))

            test_loss_list.append(test_loss)
    
    avg_test_loss = sum(test_loss_list)/len(test_loss_list)
    print(i,'parameters {} average {} Fold test loss : '.format(average_times,num_Fold),avg_test_loss)

    work_list[i] = avg_test_loss
    
    return

def FCNN_removal(Data, full_parameter, r, num_layer, num_neuron, act_f, initial, num_epochs, learning_rate, num_Fold, batch_norm, average_times, num_output, device, print_step, avg_rmse,std_rmse,avg_grad,std_grad):
    
    ##### Data Folding #####################
    print('############################ {} parameter training start #############################'.format(r))
    full_parameter.remove(r)
    parameters = list(full_parameter)
    parameters.append(Data.columns[-1])
    
    selected_Data = Data[parameters]
    shuffle_Data = selected_Data.sample(frac=1).reset_index(drop=True)
    
    try :
        Fold_length = int(len(Data)/num_Fold)
    except ZeroDivisionError :
        pass

    for j in range(num_Fold):

        if (j+1) == num_Fold:
            setattr(mod,'Fold_{}'.format(j+1),shuffle_Data.index[(j)*Fold_length:])

        else:
            setattr(mod,'Fold_{}'.format(j+1),shuffle_Data.index[(j)*Fold_length:(j+1)*Fold_length])

    ##### Network define ####################
    with tf.device(device):

        dense_layer = partial(tf.layers.dense, activation=None,kernel_initializer=getattr(tf.keras.initializers,initial)(),use_bias=True)
        training = tf.placeholder_with_default(False, shape=[], name="training")
        batch_norm_layer = partial(tf.layers.batch_normalization, training=training, momentum=0.9)

        X = tf.placeholder(tf.float32,shape=(None,len(shuffle_Data.columns)-num_output), name='X')
        Y = tf.placeholder(tf.float32,shape=(None,num_output), name='Y')

        for j in range(num_layer):

            if j == 0:
                setattr(mod, 'hidden{}'.format(j+1),dense_layer(X,num_neuron))
            else:
                setattr(mod, 'hidden{}'.format(j+1),dense_layer(getattr(mod, 'hidden{}_act'.format(j)),num_neuron))

            if batch_norm == True:
                setattr(mod,'batch_norm_hidden{}'.format(j+1),batch_norm_layer(getattr(mod,'hidden{}'.format(j+1))))
                setattr(mod,'hidden{}_act'.format(j+1),getattr(tf.nn,act_f)(getattr(mod,'batch_norm_hidden{}'.format(j+1))))
            else:
                setattr(mod,'hidden{}_act'.format(j+1),getattr(tf.nn,act_f)(getattr(mod,'hidden{}'.format(j+1))))
        
        model = dense_layer(getattr(mod,'hidden{}'.format(num_layer)), num_output)

        loss = tf.math.sqrt(tf.reduce_mean(tf.square(model - Y)))
        G_loss = tf.math.abs(tf.reduce_sum(tf.gradients(model,X)))

        if batch_norm == False:
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(mse_loss)
        else:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

    ###### training start ##########################

    with tf.Session(config=config) as sess:

        rmse = []
        grad = []

        for j in range(average_times):

            for k in range(num_Fold):
                sess.run(tf.global_variables_initializer())

                Test_X = shuffle_Data[shuffle_Data.columns[:-num_output]].iloc[getattr(mod,'Fold_{}'.format(k+1))]
                Test_Y = shuffle_Data[shuffle_Data.columns[-num_output:]].iloc[getattr(mod,'Fold_{}'.format(k+1))]
                Train_X = shuffle_Data[shuffle_Data.columns[:-num_output]].drop(Test_X.index)
                Train_Y = shuffle_Data[shuffle_Data.columns[-num_output:]].drop(Test_Y.index)

                for epoch in range(num_epochs):

                    sess.run(train_op, feed_dict={X: Train_X, Y: Train_Y, training: True})

                    if (epoch+1) % print_step == 0:
                        print('{} of {} training {} of {} Fold {} epoch '.format(j+1,average_times,k+1,num_Fold,epoch+1),
                              'loss : ',sess.run(loss, feed_dict={X: selected_Data[selected_Data.columns[:-1]], Y: selected_Data[selected_Data.columns[-1:]], training: False}),
                              'G loss : ',sess.run(G_loss, feed_dict={X: selected_Data[selected_Data.columns[:-1]], Y: selected_Data[selected_Data.columns[-1:]], training: False}))

                rmse.append(sess.run(loss, feed_dict={X: selected_Data[selected_Data.columns[:-1]], Y: selected_Data[selected_Data.columns[-1:]], training: False}))
                grad.append(sess.run(G_loss, feed_dict={X: selected_Data[selected_Data.columns[:-1]], Y: selected_Data[selected_Data.columns[-1:]], training: False}))
    
    
    print(r,'parameters {} average {} Fold test loss : '.format(average_times,num_Fold))

    avg_rmse[r] = np.mean(rmse)
    std_rmse[r] = np.std(rmse)
    avg_grad[r] = np.mean(grad)
    std_grad[r] = np.std(grad)

    return 0