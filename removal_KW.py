"""
environment :: tensorflow 1.15 version
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import numpy as np
import sys
mod = sys.modules[__name__]
from functools import partial
from multiprocessing import Process, Manager
import Network_KW

######### Options ##################################
number_of_removed_parameters = 9
num_output = 1

file_name = 'trimmed_new_case2_Re180_Pr07.csv'
file_dir = '/home/ftmlab/Downloads/wedge_code_KW/Data/'
number_of_case = 0    # no case datafile 사용한다면 0으로 설정

standardization = True
normalization = True
min_norm = 0
max_norm = 1

num_layer = 11
num_neuron = 100
act_f = 'elu'
initial = 'he_normal'
num_epochs = 30000
learning_rate = 0.00001

num_Fold = 10
batch_norm = True
average_times = 1

numb_gpu = 2
num_process = 4
print_step = 10000

removed_parameter = set([])

Full_param_rmse = 0.0014842137
Full_param_grad = 58770.633

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

########################
# del raw_Data['X']
# del raw_Data['Y']
# del raw_Data['PHI_2-Alpha_t_lsq']
# # del Data['Pr']
# del raw_Data['y_plus']

# del raw_Data["dT'T'/dx"]
# del raw_Data['S_12']
# del raw_Data['dnu_t/dx']
# del raw_Data["dT'T'/dy"]
# del raw_Data['dp_dy']
# del raw_Data['S_22']
# del raw_Data['S_11']
# del raw_Data['Mean_tfe_mol_diff']
# del raw_Data['dT/dy']
# del raw_Data['Mean_tfe_diss']
# del raw_Data['dnu_t/dy']
# del raw_Data['Mean_tke_pre_diff']
# del raw_Data['Mean_tke_mol_diff']
# del raw_Data['Mean_tke_diss']
# del raw_Data['t_var']
# del raw_Data['tke']
# del raw_Data['dp_dx']
# del raw_Data['dk/dx']
# del raw_Data['dk/dy']
########################
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

while len(removed_parameter) < number_of_removed_parameters:
    
    full_parameter = full_parameter - removed_parameter

    avg_rmse = manager.dict({i:0 for i in full_parameter})
    std_rmse = manager.dict({i:0 for i in full_parameter})
    avg_grad = manager.dict({i:0 for i in full_parameter})
    std_grad = manager.dict({i:0 for i in full_parameter})

    Network_KW.Work_manager_removal(Data, full_parameter, num_layer, num_neuron, act_f, initial, num_epochs, learning_rate, num_Fold, batch_norm, average_times, num_output, device, print_step, avg_rmse,std_rmse,avg_grad,std_grad,num_process)
    
    # J_mixed = {'{}'.format(i): 0.7*abs(avg_rmse[i]-Full_param_rmse)/avg_rmse[i] + 0.3*abs(avg_grad[i]-Full_param_grad)/avg_grad[i] for i in list(avg_rmse.keys())}
    J_mixed = {'{}'.format(i): avg_rmse[i] for i in list(avg_rmse.keys())}
    min_key = min(J_mixed.keys(), key=(lambda k: J_mixed[k]))

    R_ = {'{}'.format(i): avg_rmse[i] for i in list(avg_rmse.keys())}
    G_ = {'{}'.format(i): avg_grad[i] for i in list(avg_rmse.keys())}
    
    removed_parameter.add(min_key)

    Full_param_rmse = R_[min_key]
    Full_param_grad = G_[min_key]

    stat_csv = pd.DataFrame()
    stat_csv['index'] = [i for i in range(len(avg_rmse))]
    stat_csv['parameter'] = [i for i in list(avg_rmse.keys())]
    stat_csv['avg_rmse'] = [avg_rmse[i] for i in list(avg_rmse.keys())]
    stat_csv['std_rmse'] = [std_rmse[i] for i in list(avg_rmse.keys())]
    stat_csv['avg_grad'] = [avg_grad[i] for i in list(avg_rmse.keys())]
    stat_csv['std_grad'] = [std_grad[i] for i in list(avg_rmse.keys())]
    del stat_csv['index']
    stat_csv.to_csv(r'/home/ftmlab/Downloads/wedge_code_KW/Save/new_case2_{}L_{}N_{}removed_stat_1.csv'.format(num_layer,num_neuron,number_of_removed_parameters))

    print('#'*100)
    print('#'*100)
    print('J_mixed : ',J_mixed)
    print('#'*100)
    print('#'*100)
    print('R_ : ',R_)
    print('#'*100)
    print('#'*100)
    print('G_ : ',G_)
    print('removed parameters : ',min_key)
    print('#'*100)
    print('#'*100)