"""
environment :: tensorflow 1.15 version
"""
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
import Network_KW

######### Options ##################################
number_of_selected_parameters = 6
num_output = 1

file_name = 'Duct_bc3_Pr07.csv'
file_dir = '/home/ftmlab/Downloads/wedge_code_KW/Data/'
number_of_case = 0    # no case datafile 사용한다면 0으로 설정

Load_Data = False
load_dir = '/home/ftmlab/Downloads/wedge_code_KW/Save/'

standardization = True
normalization = True
min_norm = 0
max_norm = 1

num_layer = 7
num_neuron = 40
act_f = 'elu'
initial = 'he_normal'
num_epochs = 30000
learning_rate = 0.00001
batch_size = 9999999

limit_time = 60*60*47
num_Fold = 10
regularization = None
batch_norm = True
patience_step = 999999
average_times = 1

numb_gpu = 2
num_process = 8
print_step = 10000

##### Hardware settings ############################
if numb_gpu == 0:
    device = ['/cpu:0']*1000
else:
    device = ['/gpu:{}'.format(i) for i in range(numb_gpu)]*1000

####### Data Load #################################
if number_of_case == 0:
    raw_Data = pd.read_csv(r'{}{}'.format(file_dir,file_name))
    
else:
    file_name = file_name.split('case1_')[1]
    raw_Data = pd.DataFrame()
    for i in range(number_of_case):
        setattr(mod,'case{}'.format(i+1),pd.read_csv(r'{}case{}_{}'.format(file_dir,i+1,file_name)))
        raw_Data = raw_Data.append(getattr(mod,'case{}'.format(i+1)))

# ###################################################
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

# ###################################################
print(list(raw_Data.columns))
print('Data Load complete & Data shape :',raw_Data.shape)
####### Data Preprocessing #######################
def stz(Data):
    return (Data - Data.mean())/Data.std()

def norm(Data, min_norm=-0.9, max_norm=0.9):
    return (Data - Data.min())/(Data.max() - Data.min())*(max_norm - min_norm) + min_norm

def preprocessing(Data,standardization,normalization,min_norm,max_norm):
    """
    Data : pandas dataframe type
    standardization : bool type
    normalization : bool type
    min_norm : int or float type min value of normalization
    max_norm : int or float type max value of normalization
    """
    if standardization == True:
        stz_Data = stz(Data)
    else:
        stz_Data = Data

    if normalization == True:
        norm_Data = norm(stz_Data, min_norm, max_norm)
    else:
        norm_Data = stz_Data

    return norm_Data

Data = preprocessing(raw_Data,standardization,normalization,min_norm,max_norm)
print('Data preprocessing complete')
#################################################
manager = Manager()
full_parameter = set(list(Data.columns[:-num_output]))
selected_parameter = set([])

while len(selected_parameter) < number_of_selected_parameters:

    if Load_Data == False:
    
        remain_parameter = full_parameter - selected_parameter
        work_list = manager.dict({i:0 for i in remain_parameter})

    else :
        with open('{}Save.p'.format(load_dir),'rb') as file:
            selected_parameter = pickle.load(file)
            remain_parameter = pickle.load(file)
            work_list = pickle.load(file)

        Load_Data = False
        print('Load Data complete & Load options settings off')

    Network_KW.Work_manager(work_list, remain_parameter, num_process, start_time, limit_time, Data, selected_parameter, num_layer, num_neuron, act_f, initial, num_epochs, learning_rate, batch_size, num_Fold, regularization, batch_norm, patience_step, average_times, num_output, device, print_step)
    
    min_key = min(work_list.keys(), key=(lambda k: work_list[k]))
    
    remain_parameter.remove(min_key)
    selected_parameter.add(min_key)   
    print('#'*100)
    print('#'*100)
    print('selected parameters : ',selected_parameter)
    print(work_list)
    print('#'*100)
    print('#'*100)
