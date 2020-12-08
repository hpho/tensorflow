# ======================================================================
# Code for ANN training with features
#
# 1. Training a single ANN
# 2. Finding an optimized ANN structure (for num_layer and num_neuron)
# 3. ...
#
# ======================================================================

# ======================================================================
#  header
# ======================================================================

import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
import sys
mod = sys.modules[__name__]
import os
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Process, Manager
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor

# dummy input
data_extra_test = r'tmp_dummy.csv'

# ======================================================================
#  inputs
# ======================================================================

project_name = 'Duct_bc3_pr07'
#project_name = 'train_pool_boil'


data_dir = r'Duct_bc3_Pr07.csv'
#data_dir = r'bdd6_nd.csv'
#data_extra_test = r'bdd_nd_wide.csv'

case_type = 'duct'  # 'channel',  'duct' -> def data_preprocess
# case_type = 'bubble'
#case_type = 'bubble_hokyo'
#case_type = 'pool_boil'

ml_algorithm = 'fcnn'  # 'random forest' ,'fcnn'

n_output = 1           # number of output parameters
num_layer = [5,6,7,8]        # list of hidden layer numbers
num_neuron = [20,40,60,80]      # list of neuron numbers
act_f = 'elu'         # actuation fuction, ex) 'elu', 'sigmoid' ,'tanh', reference : https://keras.io/ko/activations/ 
                       # please compare with data scaling for consistent data & activation function...
initializer = 'he_normal'  # initializer, ex) 'glorot_uniform','random_normal', 'he_normal'
reg = ['L2',0]    # regularization to prevent overfitting : L1 or L2 and coefficient. if can't use regularization, set value of coeff. 0.
batch_normalization = True  # True or False
learning_rate = 0.00001
iteration_step = 30000  # number of iterations
sgd_opt = False         # True or False
batch_size = 1000      # setting batch size if sgd_opt is False, this option is neglected.

# RF algorithm hyper parameter
num_tree = [i for i in range(10,1000,100)]
num_depth = [i for i in range(10,20,1)]

n_gpu = 1              # if n_gpu is 0, use cpu only
n_process = 2          # number of process??, when there are a lot of models to train.

# options for K-fold training (in FCNN)
n_Fold = 10
average_time = 5

# options for singe ANN training (in FCNN_single)
frac_train_data = 0.7
average_time = 3
use_data_extra_test = False         # True or False
use_training_stop_by_criterion = False
interval = 10  # if using use_training_stop_by_criterion option

if n_gpu > 0:
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

# ======================================================================
#  function definitions
# ======================================================================

def Make_Folder(project_name):
    try:
        if not(os.path.isdir('./{}'.format(project_name))):
            os.makedirs(os.path.join('./{}'.format(project_name)))
            os.makedirs(os.path.join('./{}/model'.format(project_name)))
            os.makedirs(os.path.join('./{}/history_log'.format(project_name)))
            os.makedirs(os.path.join('./{}/result'.format(project_name)))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print('Failed to create directory')
            raise

def work_manager(data, conv_info, num_layer,num_neuron, act_f, initializer, reg, batch_normalization,learning_rate,iteration_step,batch_size, n_input, n_output, n_Fold, average_time, frac_train_data, project_name,device,avg_rmse,std_rmse, n_process, use_data_extra_test,use_training_stop_by_criterion,interval):

    tmp = 0
    procs = []
    for i in num_layer:
        for j in num_neuron:
            
            tmp += 1
            if len(num_layer)*len(num_neuron) == 1:
                p = Process(target=FCNN_single, args=(data, conv_info, \
                            i,j, \
                            act_f, initializer, reg, batch_normalization,learning_rate,iteration_step,batch_size, \
                            n_input, n_output, \
                            n_Fold, average_time, \
                            frac_train_data, \
                            project_name, device[tmp], \
                            avg_rmse,std_rmse, use_data_extra_test,use_training_stop_by_criterion,interval))
            else:
                p = Process(target=FCNN, args=(data,i,j, \
                            act_f, initializer, reg, batch_normalization,learning_rate,iteration_step,batch_size, \
                            n_input, n_output, \
                            n_Fold, average_time, \
                            frac_train_data, \
                            project_name, device[tmp], \
                            avg_rmse,std_rmse,use_training_stop_by_criterion,interval))
            procs.append(p)
            p.start()

            if len(procs) % n_process == 0:
                for p in procs:
                    p.join()

    for p in procs:
        p.join()

    return 0

def FCNN(data,num_layer,num_neuron, act_f, initializer, reg, batch_normalization,learning_rate,iteration_step,batch_size,n_input, n_output, n_Fold, average_time, frac_train_data, project_name,device,avg_rmse,std_rmse,use_training_stop_by_criterion,interval):
    """
    data : dataframe
    num_layer : int
    num_neuron : int
    act_f : str  'elu', 'sigmoid'
    initializer : str 'glorot_uniform','random_normal', 'he_normal'
    reg : list ['L2',0.01]
    batch_normalization : bool True, False
    iteration_step : int
    batch_size : int
    average_time : int
    device : str '/cpu:0','/gpu:0','/gpu:1'
    avg_rmse : manager.dict
    std_rmse : manager.dict
    """
    # ANN structure
    def fcnn():
        with tf.device(device):
            model = models.Sequential()
            model.add(layers.Dense(num_neuron, activation=act_f, input_shape=(n_input,), kernel_initializer=initializer, kernel_regularizer=getattr(tf.keras.regularizers,reg[0])(reg[1])))
            if batch_normalization :
                model.add(layers.BatchNormalization())
            for i in range(num_layer-1):
                model.add(layers.Dense(num_neuron, activation=act_f, kernel_initializer=initializer, kernel_regularizer=getattr(tf.keras.regularizers,reg[0])(reg[1])))
                if batch_normalization :
                    model.add(layers.BatchNormalization())
            model.add(layers.Dense(n_output))  # kwxxx
        opt = tf.keras.optimizers.Adam(lr=learning_rate)
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20)
        model.compile(optimizer=opt,loss='mse',metrics=[tf.keras.metrics.RootMeanSquaredError()])
        return model

    # K-fold iteration

    if use_training_stop_by_criterion:

        for i in range(average_time):

            train = data.sample(frac=frac_train_data)
            test = data.drop(train.index)
            train_target = train.pop(train.columns[-n_output])
            test_target = test.pop(test.columns[-n_output])

            val_rmse = []

            setattr(mod,'model{}'.format(i+1),fcnn())
            hist = {'loss':[],'val_loss':[]}
            for iv in range(int(iteration_step/interval)):
                interval_train,interval_test = [],[]

                for step in range(interval):
                    history = getattr(mod,'model{}'.format(i+1)).fit(train, train_target,validation_data=(test, test_target),epochs=1,verbose=0, batch_size=batch_size)
                    interval_train.append(history.history['root_mean_squared_error'][0])
                    if len(test)>0:
                        interval_test.append(history.history['val_root_mean_squared_error'][0])

                
                hist['loss'].append(sum(interval_train)/interval)
                if len(test) > 0:
                    hist['val_loss'].append(sum(interval_test)/interval)

                if abs(interval_train[-1] - interval_train[-2])/(sum(interval_train)/interval) > 0.3:
                    print('criterion #1 is satisfied')
                    break

                if iv > 0:
                    if abs(hist['loss'][-1] - hist['loss'][-2])/hist['loss'][-1] < 0.02:
                        print('criterion #2 is satisfied')
                        break

                if interval_test[-1] < 0.002:
                    print('criterion #3 is satisfied')
                    break
                if iv==int(iteration_step/interval)-1:
                    print('criterion #4 is satisfied')

            val_rmse.append(interval_test[-1])
    
    else:

        data = data.sample(frac=1).reset_index(drop=True)
        cv = KFold(n_splits=n_Fold)
        val_rmse = []
        for i,(t,v) in enumerate(cv.split(data)):
            train = data.iloc[t]
            val = data.iloc[v]
            for kk in range(average_time):
                # main training process
                setattr(mod,'model{}'.format(kk+1),fcnn())
                hist = getattr(mod,'model{}'.format(kk+1)).fit(train[train.columns[:-n_output]], train[train.columns[-n_output:]],validation_data=(val[val.columns[:-n_output]], val[val.columns[-n_output:]]),epochs=iteration_step,verbose=0, batch_size=batch_size)
                aa = pd.DataFrame()
                aa['epoch'] = [(i+1) for i in range(len(hist.history['loss']))]
                aa['rmse'] = hist.history['root_mean_squared_error']
                aa['val_rmse'] = hist.history['val_root_mean_squared_error']
                aa.to_csv(r'./{}/history_log/{}L_{}N_{}_{}Fold_{}.csv'.format(project_name,num_layer,num_neuron,i+1,n_Fold,kk+1),header=True,index=False)

                getattr(mod,'model{}'.format(kk+1)).save(r'./{}/model/{}L_{}N_{}_{}Fold_{}.h5'.format(project_name,num_layer,num_neuron,i+1,n_Fold,kk+1))
                val_rmse.append(hist.history['val_root_mean_squared_error'][-1])
                tf.keras.backend.clear_session()

    # post-process

    print('mean : ',np.mean(val_rmse),'std : ',np.std(val_rmse))
    avg_rmse['{}L_{}N'.format(num_layer,num_neuron)] = np.mean(val_rmse)
    std_rmse['{}L_{}N'.format(num_layer,num_neuron)] = np.std(val_rmse)
    return 0

def FCNN_single(data,conv_info,num_layer,num_neuron, act_f, initializer, reg, batch_normalization,learning_rate,iteration_step,batch_size,n_input, n_output, n_Fold, average_time, frac_train_data, project_name,device,avg_rmse,std_rmse, use_data_extra_test,use_training_stop_by_criterion,interval):
    """
    data : dataframe
    conv_info : dataframe
    num_layer : int
    num_neuron : int
    act_f : str  'elu', 'sigmoid'  # kwxxx
    initializer : str 'glorot_uniform','random_normal', 'he_normal'
    reg : list ['L2',0.01]
    batch_normalization : bool True, False
    iteration_step : int
    batch_size : int
    average_time : int
    device : str '/cpu:0','/gpu:0','/gpu:1'
    #device : str
    avg_rmse : manager.dict
    std_rmse : manager.dict
    """
    
    # ANN structure
    def fcnn():
        with tf.device(device):
            model = models.Sequential()
            model.add(layers.Dense(num_neuron, activation=act_f, input_shape=(n_input,), kernel_initializer=initializer, kernel_regularizer=getattr(tf.keras.regularizers,reg[0])(reg[1])))
            if batch_normalization :
                model.add(layers.BatchNormalization())
            for i in range(num_layer-1):
                model.add(layers.Dense(num_neuron, activation=act_f, kernel_initializer=initializer, kernel_regularizer=getattr(tf.keras.regularizers,reg[0])(reg[1])))
                if batch_normalization :
                    model.add(layers.BatchNormalization())
            model.add(layers.Dense(n_output))  # kwxxx
        opt = tf.keras.optimizers.Adam(lr=learning_rate)
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20)
        model.compile(optimizer=opt,loss='mse',metrics=[tf.keras.metrics.RootMeanSquaredError()])
        return model
    # main training process

    train = data.sample(frac=frac_train_data)
    test = data.drop(train.index)
    train_target = train.pop(train.columns[-n_output])
    test_target = test.pop(test.columns[-n_output])

    if use_training_stop_by_criterion:
        for i in range(average_time):
            setattr(mod,'model{}'.format(i+1),fcnn())
            hist = {'loss':[],'val_loss':[]}
            for iv in range(int(iteration_step/interval)):
                interval_train,interval_test = [],[]

                for step in range(interval):
                    history = getattr(mod,'model{}'.format(i+1)).fit(train, train_target,validation_data=(test, test_target),epochs=1,verbose=0, batch_size=batch_size)
                    interval_train.append(history.history['root_mean_squared_error'][0])
                    if len(test)>0:
                        interval_test.append(history.history['val_root_mean_squared_error'][0])

                
                hist['loss'].append(sum(interval_train)/interval)
                if len(test) > 0:
                    hist['val_loss'].append(sum(interval_test)/interval)

                if abs(interval_train[-1] - interval_train[-2])/(sum(interval_train)/interval) > 0.3:
                    print('criterion #1 is satisfied')
                    break

                if iv > 0:
                    if abs(hist['loss'][-1] - hist['loss'][-2])/hist['loss'][-1] < 0.02:
                        print('criterion #2 is satisfied')
                        break

                if interval_test[-1] < 0.002:
                    print('criterion #3 is satisfied')
                    break

                if iv==int(iteration_step/interval)-1:
                    print('criterion #4 is satisfied')

            setattr(mod,'train_pred_{}'.format(i+1),getattr(mod,'model{}'.format(i+1)).predict(train).flatten())
            if len(test) > 0:
                setattr(mod,'test_pred_{}'.format(i+1),getattr(mod,'model{}'.format(i+1)).predict(test).flatten())
            getattr(mod,'model{}'.format(i+1)).save(r'./{}/model/model{}.h5'.format(project_name,i+1))

            history_log = plt.figure()
            history_log = plt.plot([j+1 for j in range(len(hist['loss']))],hist['loss'],label='train')
            if len(test) > 0:
                history_log = plt.plot([j+1 for j in range(len(hist['loss']))],hist['val_loss'],label='val')
            history_log = plt.grid(True)
            history_log = plt.legend()
            history_log = plt.savefig(r'./{}/history_log/history{}.png'.format(project_name,i+1),dpi=300)
            history_log = plt.close()

    else:
        for i in range(average_time):
            
            model = fcnn()
            hist = model.fit(train,train_target,validation_data=(test, test_target),epochs=iteration_step,verbose=1, batch_size=batch_size)
            setattr(mod,'train_pred_{}'.format(i+1),model.predict(train).flatten())
            if len(test) > 0:
                setattr(mod,'test_pred_{}'.format(i+1),model.predict(test).flatten())

            model.save(r'./{}/model/model{}.h5'.format(project_name,i+1))

            history_log = pd.DataFrame()
            history_log['epoch'] = [(i+1) for i in range(len(hist.history['loss']))]
            history_log['rmse'] = hist.history['root_mean_squared_error']
            if len(test) > 0:
                history_log['val_rmse'] = hist.history['val_root_mean_squared_error']
            else:
                history_log['val_rmse'] = 0.0
            history_log.to_csv(r'./{}/history_log/history{}.csv'.format(project_name,i+1),header=True,index=False)

            history_log_plot = plt.figure()
            history_log_plot = plt.plot(history_log['epoch'],history_log['rmse'],label='train')
            history_log_plot = plt.plot(history_log['epoch'],history_log['val_rmse'],label='test')
            history_log_plot = plt.grid(True)
            history_log_plot = plt.legend()
            history_log_plot = plt.xlabel('epoch')
            history_log_plot = plt.ylabel('rmse')
            history_log_plot = plt.savefig(r'./{}/history_log/history{}.png'.format(project_name,i+1),dpi=300)

    
    
    # full scatter plot

    if n_output == 1 :

        
        full_scatter_plot = plt.figure()
        full_scatter_plot = plt.gca().set_aspect('equal')
        train_full_df = pd.DataFrame()
        train_full_df['actual'] = scale_inverse(train_target, conv_info, All_Data.columns[-1])
        if len(test)>0:
            test_full_df = pd.DataFrame()
            test_full_df['actual'] = scale_inverse(test_target, conv_info, All_Data.columns[-1])
        for i in range(average_time):
            full_scatter_plot = plt.scatter(scale_inverse(train_target, conv_info, All_Data.columns[-1]), \
                                            scale_inverse(getattr(mod,'train_pred_{}'.format(i+1)), conv_info, All_Data.columns[-1]),s=10,color='blue')
            train_full_df['train{}'.format(i+1)] = scale_inverse(getattr(mod,'train_pred_{}'.format(i+1)), conv_info, All_Data.columns[-1])
            if len(test) > 0:
                full_scatter_plot = plt.scatter(scale_inverse(test_target, conv_info, All_Data.columns[-1]),  \
                                                scale_inverse(getattr(mod,'test_pred_{}'.format(i+1)), conv_info, All_Data.columns[-1]),s=10,color='red')
                test_full_df['test{}'.format(i+1)] = scale_inverse(getattr(mod,'test_pred_{}'.format(i+1)), conv_info, All_Data.columns[-1])
        
        train_full_df.to_csv(r'./{}/result/full_train_data.csv'.format(project_name),header=True,index=False)
        if len(test) > 0:
            test_full_df.to_csv(r'./{}/result/full_test_data.csv'.format(project_name),header=True,index=False)
        
        x_min = scale_inverse(train_target.min(), conv_info, All_Data.columns[-1])
        x_max = scale_inverse(train_target.max(), conv_info, All_Data.columns[-1])
        
        full_scatter_plot = plt.plot([x_min,x_max],[x_min,x_max],color='k',linewidth=1.5)
        full_scatter_plot = plt.axis([x_min,x_max,x_min,x_max])
        full_scatter_plot = plt.legend(['y=x','train','test'])
        full_scatter_plot = plt.grid(True)
        full_scatter_plot = plt.xlabel('${}$ actual'.format(All_Data.columns[-1]))
        full_scatter_plot = plt.ylabel('${}$ pred'.format(All_Data.columns[-1]))
        full_scatter_plot = plt.savefig(r'./{}/result/full_scatter.png'.format(project_name),dpi=300)
        full_scatter_plot = plt.close()

        avg_scatter_plot = plt.figure()
        avg_scatter_plot = plt.gca().set_aspect('equal')

        train_avg_df = pd.DataFrame()
        train_df = pd.DataFrame([getattr(mod,'train_pred_{}'.format(i+1)) for i in range(average_time)])
        mean_train_df = train_df.mean()

        avg_scatter_plot = plt.scatter(scale_inverse(train_target, conv_info, All_Data.columns[-1]), \
                                       scale_inverse(mean_train_df, conv_info, All_Data.columns[-1]),s=10,color='blue',label='train')

        train_avg_df['actual'] = scale_inverse(train_target, conv_info, All_Data.columns[-1])
        train_avg_df['avg_model'] = train_df.mean()
        train_avg_df.to_csv(r'./{}/result/avg_train_data.csv'.format(project_name),header=True,index=False)

        if len(test) > 0:
            test_avg_df = pd.DataFrame()
            test_df = pd.DataFrame([getattr(mod,'test_pred_{}'.format(i+1)) for i in range(average_time)])
            mean_test_df = test_df.mean()
            test_avg_df['actual'] = scale_inverse(test_target, conv_info, All_Data.columns[-1])
            test_avg_df['avg_model'] = test_df.mean()
            test_avg_df.to_csv(r'./{}/result/avg_test_data.csv'.format(project_name),header=True,index=False)
    
            avg_scatter_plot = plt.scatter(scale_inverse(test_target, conv_info, All_Data.columns[-1]),  \
                                        scale_inverse(mean_test_df, conv_info, All_Data.columns[-1]),s=10,color='red',label='test')

        avg_scatter_plot = plt.plot([x_min,x_max],[x_min,x_max],color='k',linewidth=1.5)
        avg_scatter_plot = plt.axis([x_min,x_max,x_min,x_max])
        avg_scatter_plot = plt.xlabel('${}$ actual'.format(All_Data.columns[-1]))
        avg_scatter_plot = plt.ylabel('${}$ pred'.format(All_Data.columns[-1]))
        avg_scatter_plot = plt.legend()
        avg_scatter_plot = plt.grid(True)
        avg_scatter_plot = plt.savefig(r'./{}/result/avg_scatter.png'.format(project_name),dpi=300)
        avg_scatter_plot = plt.close()

    if use_data_extra_test:
        All_Data_extra_test = pd.read_csv(r'./tmpxxx_extra_test.csv')
        result_extra_test = pd.DataFrame()
        result_extra_test['pred'] = scale_inverse(model.predict(All_Data_extra_test[All_Data.columns[:-1]]).flatten(), conv_info, All_Data.columns[-1])
        result_extra_test.to_csv(r'./{}/result/result_extra_test.csv'.format(project_name),header=True,index=False)


    avg_rmse['{}L_{}N'.format(num_layer,num_neuron)] = 0.0
    std_rmse['{}L_{}N'.format(num_layer,num_neuron)] = 0.0
    return 0

def stz(Data):
    return (Data - Data.mean())/Data.std()

def stz_conv(Data, Conv_info):
    Conv_info.loc['for_a',:] = 1/Data.std()
    Conv_info.loc['for_b',:] = - Data.mean()/Data.std()
    Conv_info.loc['rev_a',:] = 1/Conv_info.loc['for_a',:]
    Conv_info.loc['rev_b',:] = -1*Conv_info.loc['for_b',:]/Conv_info.loc['for_a',:]
    return (Data - Data.mean())/Data.std()

def norm(Data, min_norm=-0.9, max_norm=0.9):
    return (Data - Data.min())/(Data.max() - Data.min())*(max_norm - min_norm) + min_norm

def norm_conv(Data, Conv_info, min_norm=-0.9, max_norm=0.9):
    Conv_info.loc['for_a',:] = 1/(Data.max() - Data.min())*(max_norm - min_norm)
    Conv_info.loc['for_b',:] = min_norm - Data.min()/(Data.max() - Data.min())*(max_norm - min_norm)
    Conv_info.loc['rev_a',:] = 1/Conv_info.loc['for_a',:]
    Conv_info.loc['rev_b',:] = -1*Conv_info.loc['for_b',:]/Conv_info.loc['for_a',:]
    return (Data - Data.min())/(Data.max() - Data.min())*(max_norm - min_norm) + min_norm

def scale_inverse(Data, Conv_info, Col):
    return Data*Conv_info.loc['rev_a',Col] + Conv_info.loc['rev_b',Col]

def scale_forward(Data, Conv_info, Col):
    return Data*Conv_info.loc['for_a',Col] + Conv_info.loc['for_b',Col]

def scale_forward_all(Data, Conv_info):
    return Data*Conv_info.loc['for_a',:] + Conv_info.loc['for_b',:]

def init_conv_info(Data):
    conv_info = pd.DataFrame(0,index=['for_a','for_b','rev_a','rev_b'], columns=Data.columns)
    conv_info.loc['for_a',:] = 1
    conv_info.loc['rev_a',:] = 1
    return conv_info

def del_column(var_name, data):
    if var_name in data.columns: 
        del data[var_name]
    return data

def data_preprocess(Data,Conv_info,type,use_data_extra_test,data_extra_test):
    """
    Data : DataFrame
    Conv_info : DataFrame
    type : str , 'bubble','bubble_hokyo','channel','duct','wedge','pool_boil'
    """

    #! for temporary use...
    Conv_info_stz = init_conv_info(Data)
    Conv_info_nor = init_conv_info(Data)
    
    if type == 'bubble':
        # note that it assumes tanh or sigmoid
        stz_Data = stz_conv(Data,Conv_info_stz)
        norm_Data = norm_conv(Data,Conv_info_nor)
        All_Data = pd.DataFrame()
        All_Data[Data.columns[0:3]] = norm_Data[Data.columns[0:3]]
        All_Data[Data.columns[3]] = stz_Data[Data.columns[3]]
        All_Data[Data.columns[4]] = norm_Data[Data.columns[4]]
        All_Data[Data.columns[5]] = stz_Data[Data.columns[5]]
        Conv_info[Data.columns[5]] = Conv_info_stz[Data.columns[5]]
        
    if type == 'bubble_hokyo':
        # note that the data is normalized for tanh or sigmoid
        All_Data = pd.DataFrame()
        All_Data = Data

    if type == 'pool_boil':
        #stz_Data = stz(Data)
        #norm_Data = norm(Data)
        #All_Data = pd.DataFrame()
        #???
        #All_Data[Data.columns[0:2]] = norm_Data[Data.columns[0:2]] 
        # note that the data is normalized for tanh or sigmoid
        #All_Data = norm(Data)
        
        #All_Data = norm_conv(Data,Conv_info)
        All_Data = norm_conv(Data,Conv_info,-0.95,0.95)
        print(Conv_info)

        if use_data_extra_test:
            Data_extra_test = pd.read_csv(r'{}'.format(data_extra_test))
            All_Data_extra_test = scale_forward_all(Data_extra_test, Conv_info)
            All_Data_extra_test.to_csv(r'./tmpxxx_extra_test.csv',header=True,index=False)

    if type == 'channel':
        # Delete unnecessary variables
        Data = del_column('X', Data)
        Data = del_column('Y', Data)
        Data = del_column('y_plus', Data)
        Data = del_column('wedge_height', Data)
        Data = del_column('dk/dx', Data)
        Data = del_column('dk/dy', Data)
        #Data = del_column('dk_dx_i_abs', Data)
        Data = del_column('dp_dx', Data)
        Data = del_column('dp_dy', Data)
        #Data = del_column('dp_dx_i_abs', Data)
        Data = del_column('u_du/dx+v_dv/dx', Data)
        Data = del_column('dT/dx', Data)
        Data = del_column('dT/dy', Data)
        #Data = del_column('dT/dx_i_abs', Data)
        Data = del_column('S_12(=S_21)', Data)
        #Data = del_column('S_ij_abs', Data)
        Data = del_column('S_11', Data)
        Data = del_column('S_22', Data)
        Data = del_column('Ui_AVG_n', Data)
        Data = del_column('Vi_AVG_n', Data)
        Data = del_column('Wi_AVG_n', Data)
        Data = del_column('du/dx', Data)
        Data = del_column('du/dy', Data)
        Data = del_column('dv/dx', Data)
        Data = del_column('dv/dy', Data)
        Data = del_column('P_AVG_n', Data)
        Data = del_column('1/Pr', Data)
        Data = del_column('PHI-Rp', Data)
        Data = del_column('PHI-Rd', Data)
        Data = del_column('PHI-Rp/Rd', Data)
        Data = del_column('tke_sdm_n', Data)
        Data = del_column('PHI-tfe_sdm_n', Data)
        Data = del_column('local_volume', Data)
        
        # note that the data is normalized for elu or relu : kwxxx
        #All_Data = norm(stz(Data),0,1)
        All_Data = norm_conv(Data,Conv_info,0,1)

    if type == 'wedge':
        # Delete unnecessary variables
        Data = del_column('X', Data)
        Data = del_column('Y', Data)
        Data = del_column('y_plus', Data)
        Data = del_column('PHI-Alpha_t_lsq', Data)
        Data = del_column('no_volume_cc', Data)
        # Data = del_column('dk/dx', Data)
        # Data = del_column('dk/dy', Data)
        #Data = del_column('dk_dx_i_abs', Data)
        # Data = del_column('dp_dx', Data)
        # Data = del_column('dp_dy', Data)
        #Data = del_column('dp_dx_i_abs', Data)
        # Data = del_column('u_du/dx+v_dv/dx', Data)
        # Data = del_column('dT/dx', Data)
        # Data = del_column('dT/dy', Data)
        #Data = del_column('dT/dx_i_abs', Data)
        # Data = del_column('S_12(=S_21)', Data)
        #Data = del_column('S_ij_abs', Data)
        # Data = del_column('S_11', Data)
        # Data = del_column('S_22', Data)
        # Data = del_column('Ui_AVG_n', Data)
        # Data = del_column('Vi_AVG_n', Data)
        # Data = del_column('Wi_AVG_n', Data)
        # Data = del_column('du/dx', Data)
        # Data = del_column('du/dy', Data)
        # Data = del_column('dv/dx', Data)
        # Data = del_column('dv/dy', Data)
        # Data = del_column('P_AVG_n', Data)
        # Data = del_column('1/Pr', Data)
        # Data = del_column('PHI-Rp', Data)
        # Data = del_column('PHI-Rd', Data)
        # Data = del_column('PHI-Rp/Rd', Data)
        # Data = del_column('tke_sdm_n', Data)
        # Data = del_column('PHI-tfe_sdm_n', Data)
        # Data = del_column('local_volume', Data)
        Data = del_column('Rp',Data)
        Data = del_column('Rd',Data)
        
        # note that the data is normalized for elu or relu
        #All_Data = norm(stz(Data),0,1)
        All_Data = norm_conv(Data,Conv_info,0,1)

    if type == 'duct':
        
        # Delete unnecessary variables
        del Data['Y']
        del Data['Z']
        del Data['Nu']
        del Data['Alpha']
        del Data['Delta']
        del Data['Alpha_t']
        del Data['dk/dz']
        del Data['dk/dy']
        del Data['S_22']
        del Data['S_33']
        del Data['S_23']
        del Data['S_32']
        del Data['S_ij_abs']
        del Data['dT/dz']
        del Data['dT/dy']
        del Data['Rp']
        del Data['Rd']
        del Data['tke_sdm_']
        del Data['tke_conv_turb_']
        del Data['TV_sdm']
        del Data['TV_conv_turb']
        del Data['dp_dz']
        del Data['dp_dy']
        del Data['w_dw/dz+v_dv/dz']
        del Data['Rp/Rd']
        del Data['U-AVG-X']
        del Data['U-AVG-Y']
        del Data['U-AVG-Z']
        del Data['dw/dz']
        del Data['dw/dy']
        del Data['dv/dz']
        del Data['dv/dy']
        del Data['P_AVG_']
        del Data['T_AVG']
        del Data['local_volume']
        del Data['local_area']
        del Data['dT/dx_i_abs']
        del Data['dp_dx_i_abs']
        del Data['dTvar/dx_i_abs']
        del Data['y_plus']
        del Data['Pr']

        # note that the data is normalized for elu or relu
        #All_Data = norm(stz(Data),0,1)
        All_Data = norm_conv(Data,Conv_info,0,1)

    #Corr matrix
    corr_mat = All_Data[All_Data.columns[0:]].corr()
    print(' ')
    print('Correlation matrix')
    print(corr_mat)

    return All_Data

# ======================================================================
#  main procedure of the program
# ======================================================================

# -------------------------------------------------
#  read file
# -------------------------------------------------

Make_Folder(project_name)

Data = pd.read_csv(r'{}'.format(data_dir))

normal_conv_info = init_conv_info(Data)
#print(normal_conv_info)
   
# -------------------------------------------------
#  initialization
# -------------------------------------------------

if n_gpu == 0:
    device = ['/cpu:0']*1000
#    device = '/cpu:0'
else:
    device = ['/gpu:{}'.format(i) for i in range(n_gpu)]*1000
#    device = '/gpu:0'

if sgd_opt:
    pass
else:
    batch_size = len(Data)

All_Data = data_preprocess(Data,normal_conv_info,case_type,use_data_extra_test,data_extra_test)
n_input = len(Data.columns) - n_output

# -------------------------------------------------
#  iterate over combinations of neurons and layers
# -------------------------------------------------

work_list = ['{}L_{}N'.format(i,j) for i in num_layer for j in num_neuron]

if __name__=='__main__':

    if ml_algorithm == 'fcnn':
        manager = Manager()
        avg_rmse = manager.dict({i:0 for i in work_list})
        std_rmse = manager.dict({i:0 for i in work_list})

        work_manager(All_Data, normal_conv_info, \
        num_layer, num_neuron, \
        act_f, initializer, reg, batch_normalization,learning_rate,iteration_step,batch_size, \
        n_input, n_output, \
        n_Fold, average_time, \
        frac_train_data, \
        project_name, device, \
        avg_rmse,std_rmse, n_process, use_data_extra_test,use_training_stop_by_criterion,interval)

        min_key = min(avg_rmse.keys(), key=(lambda k: avg_rmse[k]))

        print('min value : ',min_key,avg_rmse[min_key])

        result_csv = pd.DataFrame()
        result_csv['avg_rmse'] = avg_rmse.values()
        result_csv['std_rmse'] = std_rmse.values()
        result_csv.index = avg_rmse.keys()
        result_csv.to_csv(r'./{}/result/stat.csv'.format(project_name),header=True,index=True)

    if ml_algorithm == 'random forest':

        if n_Fold == 1:
            train = All_Data.sample(frac = frac_train_data)
            test  = All_Data.drop(train.index)
            train_target = train.pop(All_Data.columns[-1])
            test_target = test.pop(All_Data.columns[-1])

            score = []
            for i in num_tree:
                for j in num_depth:
                    model = RandomForestRegressor(n_estimators=i, criterion="mse", max_depth=j)
                    model.fit(train,train_target)
                    model.score(train,train_target)



        else:
            hyper_parameter = {'{}_tree_{}_depth'.format(i,j):0 for i in num_tree for j in num_depth}
            for i in num_tree:
                for j in num_depth:
                    model = RandomForestRegressor(n_estimators=i, criterion="mse", max_depth=j)
                    kfold = KFold(n_splits=n_Fold, shuffle=True, random_state=None)
                    results = cross_val_score(model, All_Data[All_Data.columns[:-n_output]],All_Data[All_Data.columns[-n_output:]], cv=kfold)
                    print(hyper_parameter)
                    hyper_parameter['{}_tree_{}_depth'.format(i,j)] = np.mean(results)

            num_tree = int(min(hyper_parameter.keys()).split('_')[0])
            num_depth = int(min(hyper_parameter.keys()).split('_')[2])

            model = RandomForestRegressor(n_estimators=num_tree, criterion="mse", max_depth=num_depth)
            y_pred = cross_val_predict(model, All_Data[All_Data.columns[:-n_output]],All_Data[All_Data.columns[-n_output:]], cv=kfold)

            y_pred_conv = scale_inverse(y_pred, conv_info, All_Data.columns[-1])
            y_actual_conv = scale_inverse(All_Data[All_Data.columns[-1]], conv_info, All_Data.columns[-1])
            
            min_conv = scale_inverse(All_Data[All_Data.columns[-1]].min(), conv_info, All_Data.columns[-1])
            max_conv = scale_inverse(All_Data[All_Data.columns[-1]].max(), conv_info, All_Data.columns[-1])

            scatter = plt.gca().set_aspect('euqal')
            scatter = plt.scatter(y_actual_conv, y_pred_conv, s=10, color='blue')
            scatter = plt.plot([min_conv,max_conv],[min_conv,max_conv], color='k', linewidth=1.5)
            scatter = plt.axis([min_conv,max_conv,min_conv,max_conv])
            scatter = plt.grid(True)
            scatter = plt.savefig(r'./{}/result/RF_model_scatter_plot.png',dpi=300)
            scatter = plt.close()

