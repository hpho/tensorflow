# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import tensorflow.compat.v1.keras.backend as KK
from tensorflow.keras import models, layers
import numpy as np
import pandas as pd
import sys
mod = sys.modules[__name__]
import os
from multiprocessing import Process, Manager
from sklearn.model_selection import KFold

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:

#   # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
#     except RuntimeError as e:
#     # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
#         print(e)

##################################################

project_name = 'Duct_bc3_Pr07'
data_dir = r'/home/ftmlab/Downloads/wedge_code_KW/Data/Duct_bc3_Pr07.csv'

n_Fold = 10    
average_time = 5
n_output = 1
num_layer = [3,5,7,9]
num_neuron = [20,40,60,80]
act_f = 'elu'
initializer = 'he_normal'
reg = ['L2',0.0001]
batch_normalization = True
learning_rate = 0.0001
iteration_step = 30000
batch_size = 1000
numb_gpu = 2
n_process = 4

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

def work_manager(data,num_layer,num_neuron,average_time,act_f, initializer, reg, batch_normalization,learning_rate,iteration_step,batch_size,n_input, n_output, n_Fold, project_name,device,avg_rmse,std_rmse, n_process):

    tmp = 0
    procs = []
    for i in num_layer:
        for j in num_neuron:
            
            p = Process(target=FCNN, args=(data,i,j,average_time,act_f, initializer, reg, batch_normalization,learning_rate,iteration_step,batch_size,n_input, n_output, n_Fold, project_name,device[tmp],avg_rmse,std_rmse))
            procs.append(p)
            p.start()
            tmp += 1

            if len(procs) % n_process == 0:
                for p in procs:
                    p.join()

    for p in procs:
        p.join()

    return 0


def FCNN(data,num_layer,num_neuron,average_time,act_f, initializer, reg, batch_normalization,learning_rate,iteration_step,batch_size,n_input, n_output, n_Fold, project_name,device,avg_rmse,std_rmse):
    """
    data : dataframe
    num_layer : int
    num_neuron : int
    average_time : int
    act_f : str  'elu', 'sigmoid'
    initializer : str 'glorot_uniform','random_normal', 'he_normal'
    reg : list ['L2',0.01]
    batch_normalization : bool True, False
    iteration_step : int
    batch_size : int
    device : str '/gpu:0','/gpu:1'
    avg_rmse : manager.dict
    std_rmse : manager.dict
    """
    def get_session(gpu_fraction=0.3):
 
        num_threads = os.environ.get('OMP_NUM_THREADS')
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    
        if num_threads:
            return tf.Session(config=tf.ConfigProto(
                gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
        else:
            return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
 
 
    KK.set_session(get_session())
    
    with tf.device(device):
        model = models.Sequential()
        model.add(layers.Dense(num_neuron, activation=act_f, input_shape=(n_input,), kernel_initializer=initializer))
        if batch_normalization :
            model.add(layers.BatchNormalization())
        for i in range(num_layer-1):
            model.add(layers.Dense(num_neuron, activation=act_f, kernel_initializer=initializer))
            if batch_normalization :
                model.add(layers.BatchNormalization())
        model.add(layers.Dense(n_output))
    opt = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=opt,loss='mse',metrics=[tf.keras.metrics.RootMeanSquaredError()])


    data = data.sample(frac=1).reset_index(drop=True)
    cv = KFold(n_splits=n_Fold)
    val_rmse = []
    for i,(t,v) in enumerate(cv.split(data)):
        train = data.iloc[t]
        val = data.iloc[v]
        for kk in range(average_time):
            hist = model.fit(train[train.columns[:-n_output]], train[train.columns[-n_output:]],validation_data=(val[val.columns[:-n_output]], val[val.columns[-n_output:]]),epochs=iteration_step,verbose=0, batch_size=batch_size)
            aa = pd.DataFrame()
            aa['epoch'] = [(i+1) for i in range(len(hist.history['loss']))]
            aa['rmse'] = hist.history['root_mean_squared_error']
            aa['val_rmse'] = hist.history['val_root_mean_squared_error']
            aa.to_csv(r'./{}/history_log/{}L_{}N_{}_{}Fold_{}.csv'.format(project_name,num_layer,num_neuron,i+1,n_Fold,kk+1),header=True,index=False)

            model.save(r'./{}/model/{}L_{}N_{}_{}Fold_{}.h5'.format(project_name,num_layer,num_neuron,i+1,n_Fold,kk+1))
            val_rmse.append(hist.history['val_root_mean_squared_error'][-1])
            tf.keras.backend.clear_session()

    print('mean : ',np.mean(val_rmse),'std : ',np.std(val_rmse))
    avg_rmse['{}L_{}N'.format(num_layer,num_neuron)] = np.mean(val_rmse)
    std_rmse['{}L_{}N'.format(num_layer,num_neuron)] = np.std(val_rmse)
    return 0

Make_Folder(project_name)
raw_Data = pd.read_csv(r'{}'.format(data_dir))
device = ['/gpu:{}'.format(i) for i in range(numb_gpu)]*1000

def stz(Data):
    return (Data - Data.mean())/Data.std()

def norm(Data, min_norm=-0.9, max_norm=0.9):
    return (Data - Data.min())/(Data.max() - Data.min())*(max_norm - min_norm) + min_norm

# duct data
del raw_Data['Y']
del raw_Data['Z']
del raw_Data['Nu']
del raw_Data['Alpha']
del raw_Data['Delta']
del raw_Data['Alpha_t']
del raw_Data['dk/dz']
del raw_Data['dk/dy']
del raw_Data['S_22']
del raw_Data['S_33']
del raw_Data['S_23']
del raw_Data['S_32']
del raw_Data['S_ij_abs']
del raw_Data['dT/dz']
del raw_Data['dT/dy']
del raw_Data['Rp']
del raw_Data['Rd']
del raw_Data['tke_sdm_']
del raw_Data['tke_conv_turb_']
del raw_Data['TV_sdm']
del raw_Data['TV_conv_turb']
del raw_Data['dp_dz']
del raw_Data['dp_dy']
del raw_Data['w_dw/dz+v_dv/dz']
del raw_Data['Rp/Rd']
del raw_Data['U-AVG-X']
del raw_Data['U-AVG-Y']
del raw_Data['U-AVG-Z']
del raw_Data['dw/dz']
del raw_Data['dw/dy']
del raw_Data['dv/dz']
del raw_Data['dv/dy']
del raw_Data['P_AVG_']
del raw_Data['T_AVG']
del raw_Data['local_volume']
del raw_Data['local_area']
del raw_Data['dT/dx_i_abs']
del raw_Data['dp_dx_i_abs']
del raw_Data['dTvar/dx_i_abs']
del raw_Data['y_plus']
del raw_Data['Pr']

n_input = len(raw_Data.columns) - 1

All_Data = norm(stz(raw_Data), 0, 1)

# bubble data
# stz_Data = stz(Data)
# norm_Data = norm(Data)

# All_Data = pd.DataFrame()
# All_Data[Data.columns[0:3]] = norm_Data[Data.columns[0:3]]
# All_Data[Data.columns[3]] = stz_Data[Data.columns[3]]
# All_Data[Data.columns[4]] = norm_Data[Data.columns[4]]
# All_Data[Data.columns[5]] = stz_Data[Data.columns[5]]

work_list = ['{}L_{}N'.format(i,j) for i in num_layer for j in num_neuron]

if __name__=='__main__':

    manager = Manager()
    avg_rmse = manager.dict({i:0 for i in work_list})
    std_rmse = manager.dict({i:0 for i in work_list})

    work_manager(All_Data,num_layer,num_neuron,average_time,act_f, initializer, reg, batch_normalization,learning_rate,iteration_step,batch_size,n_input, n_output,n_Fold,project_name,device,avg_rmse,std_rmse, n_process)

    min_key = min(avg_rmse.keys(), key=(lambda k: avg_rmse[k]))

    print('min value : ',min_key,avg_rmse[min_key])

    result_csv = pd.DataFrame()
    result_csv['avg_rmse'] = avg_rmse.values()
    result_csv['std_rmse'] = std_rmse.values()
    result_csv.index = avg_rmse.keys()
    result_csv.to_csv(r'./{}/result/stat.csv'.format(project_name),header=True,index=True)

