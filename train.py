import os
import torch
import numpy as np
import scipy.io
import models as models
import pickle
import time
import torch.nn as nn
import math
import skrf as rf
import errno
import json
import copy

from scipy.signal import savgol_filter

from common import eq_ckt

import ast
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib.pyplot as plt
plt.style.use(style='default')
plt.rcParams['font.family']='calibri'
plt.rcParams['lines.linewidth'] = 2

font = {'weight': 'bold',
        'size': 10}
plt.rc('font', **font)
plt.rc('axes', labelweight='bold')

cwd = os.getcwd()

#Device to be used in training. If GPU is available, it will be automatically used.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tcoil_data = pd.read_csv(f'~/TCoil_ML/data/gf22/train/tcoil_0.1-100.0GHz_2570_2021-10-27.csv')

tcoil_data = tcoil_data.drop_duplicates(subset=['L','W','S','Nin','Nout'])

# generate training and testing dataset, also data got shuffled 
tcoil_train, tcoil_test = train_test_split(tcoil_data, test_size = 0.2)

def csv_parser(y):
    buffer = np.empty([np.shape(y)[0], 12, 1000])
    for i in range(np.shape(y)[0]):
        
        s11 = y[i][0][1:-1].replace('(','').replace(')','').split(',')
        s12 = y[i][1][1:-1].replace('(','').replace(')','').split(',')
        s13 = y[i][2][1:-1].replace('(','').replace(')','').split(',')
        s22 = y[i][3][1:-1].replace('(','').replace(')','').split(',')
        s23 = y[i][4][1:-1].replace('(','').replace(')','').split(',')
        s33 = y[i][5][1:-1].replace('(','').replace(')','').split(',')

        buffer[i][0][:]=np.real([complex(_) for _ in s11])
        buffer[i][1][:]=np.real([complex(_) for _ in s12])
        buffer[i][2][:]=np.real([complex(_) for _ in s13])
        buffer[i][3][:]=np.real([complex(_) for _ in s22])
        buffer[i][4][:]=np.real([complex(_) for _ in s23])
        buffer[i][5][:]=np.real([complex(_) for _ in s33])
        
        buffer[i][6][:]=np.imag([complex(_) for _ in s11])
        buffer[i][7][:]=np.imag([complex(_) for _ in s12])
        buffer[i][8][:]=np.imag([complex(_) for _ in s13])
        buffer[i][9][:]=np.imag([complex(_) for _ in s22])
        buffer[i][10][:]=np.imag([complex(_) for _ in s23])
        buffer[i][11][:]=np.imag([complex(_) for _ in s33])
        

    return buffer
                 
tcoil_x_train = np.array(tcoil_train[['L','W','S','Nin','Nout']].copy())
tcoil_y_train = np.array(tcoil_train[['s11', 's12', 's13', 's22', 's23', 's33']].copy())
tcoil_y_train = csv_parser(tcoil_y_train)
                 
tcoil_x_test = np.array(tcoil_test[['L','W','S','Nin','Nout']].copy())
tcoil_y_test = np.array(tcoil_test[['s11', 's12', 's13', 's22', 's23', 's33']].copy())
tcoil_y_test = csv_parser(tcoil_y_test)

# normalize the input data   
mean_x_train = tcoil_x_train.mean(axis=0)
std_x_train = tcoil_x_train.std(axis=0)
tcoil_x_train = (tcoil_x_train-mean_x_train)/std_x_train
tcoil_x_test = (tcoil_x_test-mean_x_train)/std_x_train

# normalize the output data   
mean_y_train = tcoil_y_train.mean(axis=0)
std_y_train = tcoil_y_train.std(axis=0)
tcoil_y_train = (tcoil_y_train-mean_y_train)/std_y_train
tcoil_y_test = (tcoil_y_test-mean_y_train)/std_y_train

# save mean and std of the training data to JSON 
mean_std_train = {}
mean_std_train['input']={'mean':mean_x_train.tolist(),'std':std_x_train.tolist()}
mean_std_train['output']={'mean':mean_y_train.tolist(),'std':std_y_train.tolist()}

with open('mean_std_train.json','w') as f:
    json.dump(mean_std_train,f)

# convert DataFrame to np array so they can be fed to PyTorch model
tcoil_y_train = torch.Tensor(tcoil_y_train).to(device)
tcoil_x_train = torch.Tensor(tcoil_x_train).to(device)

tcoil_y_test = torch.Tensor(tcoil_y_test).to(device)
tcoil_x_test = torch.Tensor(tcoil_x_test).to(device)

print('Done loading data and pre-processing.')

##############################################################################
NMSE_Test = []
modelWeights= []
NMSE_Test_Median = []

model = models.Tcoil_S_UpCNN().to(device) # best since it is checkerboard artifact free

#Select optimizer to be used, define initial "learning rate (lr)", and learning rate reduction ratio (gamma) at milestones.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones= [300, 500, 1000, 2500, 3500], gamma=0.5)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones= [1500, 2000, 2500, 2750], gamma=0.5)
numParams = sum([p.numel() for p in model.parameters()])

print(f"Model is loaded. Number of learnable parameters in the network {numParams:d}")


#Training Loss Function. This is different than MSE loss, see the paper for details.
def calc_training_loss(recon_y, y):
    err = (torch.abs(y-recon_y)**2).sum(dim=2)/recon_y.shape[2]
    return err.sqrt().mean()

#Calculate Test Accuracy. For each response in the data (L(f), R(f)), calculate NMSE (see paper for details).
def calc_test_NMSE(recon_y, y):
    err = (torch.abs(y-recon_y)**2).sum(dim=2)
    mean_ref = y.mean(dim=2)
    mean_ref = mean_ref.unsqueeze(-1).repeat(1, 1, y.shape[2])
    norm = (torch.abs(y-mean_ref)**2).sum(dim=2)
    NMSE = err/norm
    return NMSE

#Closure to be called by optimizer during training.
def closure(data_x, data_y, model):
    optimizer.zero_grad()
    output = model(data_x)
    loss = calc_training_loss(output, data_y)
    return loss

#Man Training Loop.
#Results are printed at every "test_schedule" epochs.
test_schedule = 5
training_iter = 3000
current_time = time.time()
print(f"Starting training the model. \n")
print(f"""-----------------------------------------------------------------""")

for a in range(training_iter):
    model.train()
    train_data_x = tcoil_x_train
    train_data_y = tcoil_y_train
    loss = closure(train_data_x, train_data_y, model)

    loss.backward()

    optimizer.step()
    scheduler.step()

    # Set into eval mode for testing.
    if a % test_schedule == 0:
        model.eval()
        with torch.no_grad():
            test_y = tcoil_y_test
            test_x = tcoil_x_test
            test_output = model(test_x)

            NMSE = calc_test_NMSE(test_output, test_y)
            avNMSE = NMSE.mean()
            medNMSE = NMSE.median()

            print(f"Train Iter {(a+1):d}/{training_iter:d} - Train Loss: {loss.item():.3f} --> "
                  f"Test Av. NMSE: {avNMSE.item():.3f}, Med. NMSE: {medNMSE.item():.3f}")

            #save model weights at each iteration. The best model will be chosen after training is finished.
            modelWeights += [model.state_dict()]
            
            #save test NMSE (both average and median)
            NMSE_Test += [avNMSE.item()]
            NMSE_Test_Median += [medNMSE.item()]


elapsed = time.time() - current_time
print(f"""\n-----------------------------------------------------------------""")
print(f"""Training is completed in {elapsed/60 :.3f} minutes""")

###################################### validations ##########################################

NMSE_Test = np.asarray(NMSE_Test)
best_idx = np.argmin(NMSE_Test)


def s_params_parser(y, filter=False, save2touchstone=False):
    y = y.detach().numpy()
    s_params = np.array([np.array(std_y_train)*np.array(y[case])+np.array(mean_y_train) for case in range(len(y))])
    
    if filter==True:
        for i in range(np.shape(s_params)[1]):
            s_params[:,i,:] = savgol_filter(s_params[:,i,:], 49, 3)
        
    s11 = s_params[:,0,:] + s_params[:,6,:] * 1j
    s12 = s_params[:,1,:] + s_params[:,7,:] * 1j
    s13 = s_params[:,2,:] + s_params[:,8,:] * 1j
    s21 = s12
    s22 = s_params[:,3,:] + s_params[:,9,:] * 1j
    s23 = s_params[:,4,:] + s_params[:,10,:] * 1j
    s31 = s13
    s32 = s23
    s33 = s_params[:,5,:] + s_params[:,11,:] * 1j
    
    s_params = np.empty([np.shape(y)[0], np.shape(y)[2], 3, 3], dtype=np.complex128)
    for i in range(np.shape(y)[0]):
        for j in range(np.shape(y)[2]):
            s_params[i][j][0][0] = s11[i][j]
            s_params[i][j][0][1] = s12[i][j]
            s_params[i][j][0][2] = s13[i][j]
            s_params[i][j][1][0] = s21[i][j]
            s_params[i][j][1][1] = s22[i][j]
            s_params[i][j][1][2] = s23[i][j]
            s_params[i][j][2][0] = s31[i][j]
            s_params[i][j][2][1] = s32[i][j]
            s_params[i][j][2][2] = s33[i][j]
    
    La = []
    Lb = []
    Qa = []
    Qb = []
    Ra = []
    Rb = []
    k = []
    SRF= []
    
    for i in range(np.shape(y)[0]):
        s = s_params[i]
        f = np.array([i*0.1 for i in range(1, np.shape(y)[2]+1)])
        network = rf.Network(frequency=f, s=s)
        if save2touchstone == True:
            directory = f'/autofs/fs1.ece/fs1.eecg.tcc/lizongh2/S-TCNN/tcoil_pf/tcoil_pf{i}/'
            filename = f'tcoil_pf{i}.s3p'
            if not os.path.exists(os.path.dirname(directory+filename)):
                try:
                    os.makedirs(os.path.dirname(directory+filename))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            network.write_touchstone(filename=filename, dir=directory)
            # Unfortunately, sktf does not support save y-parameter to touchstone file...
            s3p2yraw(network, directory, f'tcoil_pf{i}.y')
            
        LQRk = eq_ckt.t_network_eq_ckt(network)
        La.append(LQRk['La'])
        Qa.append(LQRk['Qa'])
        Lb.append(LQRk['Lb'])
        Qb.append(LQRk['Qb'])
        Ra.append(LQRk['Ra'])
        Rb.append(LQRk['Rb'])
        k.append(-LQRk['k']) # gf22 k is negative
        SRF.append(LQRk['fr'])
        
    return {
            's_params': s_params,
            'La': La,
            'Lb': Lb,
            'Qa': Qa,
            'Qb': Qb,
            'Ra': Ra,
            'Rb': Rb,
            'k': k,
            'fr': SRF,
            }

def s3p2yraw(network, directory, filename):
    with open(directory+filename,'w') as out:
        lines = []
        y = network.y
        for i in range(len(network.f)):
            y11_real = np.real(y[i][0][0])
            y11_imag = np.imag(y[i][0][0])
            y12_real = np.real(y[i][0][1])
            y12_imag = np.imag(y[i][0][1])
            y13_real = np.real(y[i][0][2])
            y13_imag = np.imag(y[i][0][2])
            y21_real = y12_real
            y21_imag = y12_imag
            y22_real = np.real(y[i][1][1])
            y22_imag = np.imag(y[i][1][1])
            y23_real = np.real(y[i][1][2])
            y23_imag = np.imag(y[i][1][2])
            y31_real = y13_real
            y31_imag = y13_imag
            y32_real = y23_real
            y32_imag = y23_imag
            y33_real = np.real(y[i][2][2])
            y33_imag = np.imag(y[i][2][2])
            
            lines.append([f"{network.f[i]} {y11_real} {y11_imag} {y12_real} {y12_imag} {y13_real} {y13_imag} {y21_real} {y21_imag} {y22_real} {y22_imag} {y23_real} {y23_imag} {y31_real} {y31_imag} {y32_real} {y32_imag} {y33_real} {y33_imag}"])
        
        for line in lines:
            out.write(f'{line[0]}\n')

test_results = s_params_parser(test_data_y.cpu())
pred_results = s_params_parser(model(test_data_x).cpu())

# You may ask what is 'pred_eq_ckt_results'? You can open Cadence VIirtuoso with gf22, and there is a library called tcoil and a netlist called test,
# enter the tcoil geometry there and run s-parameter analysis, then you get the s3p of the t-coil equivalent circuit.

def plot(test_results, pred_results, pred_eq_ckt_results=None, case=0):
    f = np.arange(0,freq_size_sliced)/1

    plt.figure('s11_test vs. s11_pred')                                               
    fig, axs = plt.subplots(2)
    axs[0].plot(f, np.real(test_results['s_params'][case][:,0,0]), 'r')
    axs[0].plot(f, np.real(pred_results['s_params'][case][:,0,0]), 'b')        
    axs[0].set(ylabel='s11 real')
    axs[1].plot(f, np.imag(test_results['s_params'][case][:,0,0]), 'r')
    axs[1].plot(f, np.imag(pred_results['s_params'][case][:,0,0]), 'b') 
    if pred_eq_ckt_results != None:
        axs[0].plot(f, np.real(pred_eq_ckt_results['s_params'][case][:,0,0]), 'g')
        axs[1].plot(f, np.imag(pred_eq_ckt_results['s_params'][case][:,0,0]), 'g')
        axs[0].legend(('real(s11_test)', 'real(s11_pred)', 'real(s11_pred_eq_ckt)'))
        axs[1].legend(('imag(s11_test)', 'imag(s11_pred)', 'imag(s11_pred_eq_ckt)'))  
        fig.suptitle('s11_test vs. s11_pred vs s11_pred_eq_ckt')
    else:
        axs[0].legend(('real(s11_test)', 'real(s11_pred)'))
        axs[1].legend(('imag(s11_test)', 'imag(s11_pred)')) 
        fig.suptitle('s11_test vs. s11_pred')
    axs[1].set(xlabel='Frequency (GHz)') 
    axs[1].set(ylabel='s11 imag')   
    axs[0].grid()
    axs[1].grid()
    
    plt.figure('s12_test vs. s12_pred')                                               
    fig, axs = plt.subplots(2)
    axs[0].plot(f, np.real(test_results['s_params'][case][:,0,1]), 'r')
    axs[0].plot(f, np.real(pred_results['s_params'][case][:,0,1]), 'b')        
    axs[0].set(ylabel='s12 real')
    axs[1].plot(f, np.imag(test_results['s_params'][case][:,0,1]), 'r')
    axs[1].plot(f, np.imag(pred_results['s_params'][case][:,0,1]), 'b') 
    if pred_eq_ckt_results != None:
        axs[0].plot(f, np.real(pred_eq_ckt_results['s_params'][case][:,0,1]), 'g')
        axs[1].plot(f, np.imag(pred_eq_ckt_results['s_params'][case][:,0,1]), 'g')
        axs[0].legend(('real(s12_test)', 'real(s12_pred)', 'real(s12_pred_eq_ckt)'))
        axs[1].legend(('imag(s12_test)', 'imag(s12_pred)', 'imag(s12_pred_eq_ckt)'))  
        fig.suptitle('s12_test vs. s12_pred vs s12_pred_eq_ckt')
    else:
        axs[0].legend(('real(s12_test)', 'real(s12_pred)'))
        axs[1].legend(('imag(s12_test)', 'imag(s12_pred)')) 
        fig.suptitle('s12_test vs. s12_pred')
    axs[1].set(xlabel='Frequency (GHz)') 
    axs[1].set(ylabel='s12 imag')   
    axs[0].grid()
    axs[1].grid()
    

    plt.figure('s13_test vs. s13_pred')                                               
    fig, axs = plt.subplots(2)
    axs[0].plot(f, np.real(test_results['s_params'][case][:,0,2]), 'r')
    axs[0].plot(f, np.real(pred_results['s_params'][case][:,0,2]), 'b')        
    axs[0].set(ylabel='s13 real')
    axs[1].plot(f, np.imag(test_results['s_params'][case][:,0,2]), 'r')
    axs[1].plot(f, np.imag(pred_results['s_params'][case][:,0,2]), 'b') 
    if pred_eq_ckt_results != None:
        axs[0].plot(f, np.real(pred_eq_ckt_results['s_params'][case][:,0,2]), 'g')
        axs[1].plot(f, np.imag(pred_eq_ckt_results['s_params'][case][:,0,2]), 'g')
        axs[0].legend(('real(s13_test)', 'real(s13_pred)', 'real(s13_pred_eq_ckt)'))
        axs[1].legend(('imag(s13_test)', 'imag(s13_pred)', 'imag(s13_pred_eq_ckt)'))  
        fig.suptitle('s13_test vs. s13_pred vs s13_pred_eq_ckt')
    else:
        axs[0].legend(('real(s13_test)', 'real(s13_pred)'))
        axs[1].legend(('imag(s13_test)', 'imag(s13_pred)')) 
        fig.suptitle('s13_test vs. s13_pred')
    axs[1].set(xlabel='Frequency (GHz)') 
    axs[1].set(ylabel='s13 imag')   
    axs[0].grid()
    axs[1].grid()
    
    
    plt.figure('s22_test vs. s22_pred')                                               
    fig, axs = plt.subplots(2)
    axs[0].plot(f, np.real(test_results['s_params'][case][:,1,1]), 'r')
    axs[0].plot(f, np.real(pred_results['s_params'][case][:,1,1]), 'b')        
    axs[0].set(ylabel='s22 real')
    axs[1].plot(f, np.imag(test_results['s_params'][case][:,1,1]), 'r')
    axs[1].plot(f, np.imag(pred_results['s_params'][case][:,1,1]), 'b') 
    if pred_eq_ckt_results != None:
        axs[0].plot(f, np.real(pred_eq_ckt_results['s_params'][case][:,1,1]), 'g')
        axs[1].plot(f, np.imag(pred_eq_ckt_results['s_params'][case][:,1,1]), 'g')
        axs[0].legend(('real(s22_test)', 'real(s22_pred)', 'real(s22_pred_eq_ckt)'))
        axs[1].legend(('imag(s22_test)', 'imag(s22_pred)', 'imag(s22_pred_eq_ckt)'))  
        fig.suptitle('s22_test vs. s22_pred vs s22_pred_eq_ckt')
    else:
        axs[0].legend(('real(s22_test)', 'real(s22_pred)'))
        axs[1].legend(('imag(s22_test)', 'imag(s22_pred)')) 
        fig.suptitle('s22_test vs. s22_pred')
    axs[1].set(xlabel='Frequency (GHz)') 
    axs[1].set(ylabel='s22 imag')   
    axs[0].grid()
    axs[1].grid()
    
    
    plt.figure('s23_test vs. s23_pred')                                               
    fig, axs = plt.subplots(2)
    axs[0].plot(f, np.real(test_results['s_params'][case][:,1,2]), 'r')
    axs[0].plot(f, np.real(pred_results['s_params'][case][:,1,2]), 'b')        
    axs[0].set(ylabel='s23 real')
    axs[1].plot(f, np.imag(test_results['s_params'][case][:,1,2]), 'r')
    axs[1].plot(f, np.imag(pred_results['s_params'][case][:,1,2]), 'b') 
    if pred_eq_ckt_results != None:
        axs[0].plot(f, np.real(pred_eq_ckt_results['s_params'][case][:,1,2]), 'g')
        axs[1].plot(f, np.imag(pred_eq_ckt_results['s_params'][case][:,1,2]), 'g')
        axs[0].legend(('real(s23_test)', 'real(s23_pred)', 'real(s23_pred_eq_ckt)'))
        axs[1].legend(('imag(s23_test)', 'imag(s23_pred)', 'imag(s23_pred_eq_ckt)'))  
        fig.suptitle('s23_test vs. s23_pred vs s23_pred_eq_ckt')
    else:
        axs[0].legend(('real(s23_test)', 'real(s23_pred)'))
        axs[1].legend(('imag(s23_test)', 'imag(s23_pred)')) 
        fig.suptitle('s23_test vs. s23_pred')
    axs[1].set(xlabel='Frequency (GHz)') 
    axs[1].set(ylabel='s23 imag')   
    axs[0].grid()
    axs[1].grid()
    
    plt.figure('s33_test vs. s33_pred')                                               
    fig, axs = plt.subplots(2)
    axs[0].plot(f, np.real(test_results['s_params'][case][:,2,2]), 'r')
    axs[0].plot(f, np.real(pred_results['s_params'][case][:,2,2]), 'b')        
    axs[0].set(ylabel='s33 real')
    axs[1].plot(f, np.imag(test_results['s_params'][case][:,2,2]), 'r')
    axs[1].plot(f, np.imag(pred_results['s_params'][case][:,2,2]), 'b') 
    if pred_eq_ckt_results != None:
        axs[0].plot(f, np.real(pred_eq_ckt_results['s_params'][case][:,2,2]), 'g')
        axs[1].plot(f, np.imag(pred_eq_ckt_results['s_params'][case][:,2,2]), 'g')
        axs[0].legend(('real(s33_test)', 'real(s33_pred)', 'real(s33_pred_eq_ckt)'))
        axs[1].legend(('imag(s33_test)', 'imag(s33_pred)', 'imag(s33_pred_eq_ckt)'))  
        fig.suptitle('s33_test vs. s33_pred vs s33_pred_eq_ckt')
    else:
        axs[0].legend(('real(s33_test)', 'real(s33_pred)'))
        axs[1].legend(('imag(s33_test)', 'imag(s33_pred)')) 
        fig.suptitle('s33_test vs. s33_pred')
    axs[1].set(xlabel='Frequency (GHz)') 
    axs[1].set(ylabel='s33 imag')   
    axs[0].grid()
    axs[1].grid()
    
                
    plt.figure('L_test vs. L_pred')                                               
    fig, axs = plt.subplots(2)
    axs[0].plot(f, test_results['La'][case], 'r')
    axs[0].plot(f, pred_results['La'][case], 'b')       
    axs[0].set(ylabel='La (H)')
    axs[0].legend(('La_test', 'La_pred'))
    axs[1].plot(f, test_results['Lb'][case], 'r')
    axs[1].plot(f, pred_results['Lb'][case], 'b')  
    axs[1].set(ylabel='Lb (H)')   
    axs[1].set(xlabel='Frequency (GHz)')   
    if pred_eq_ckt_results != None:
        axs[0].plot(f, pred_eq_ckt_results['La'][case], 'g')
        axs[1].plot(f, pred_eq_ckt_results['Lb'][case], 'g')
        axs[0].legend(('La_test', 'La_pred', 'La_pred_eq_ckt'))
        axs[1].legend(('Lb_test', 'Lb_pred', 'Lb_pred_eq_ckt'))
        fig.suptitle('L_test vs. L_pred vs. L_pred_eq_ckt')
    else:
        axs[0].legend(('La_test', 'La_pred'))
        axs[1].legend(('Lb_test', 'Lb_pred'))                     
        fig.suptitle('L_test vs. L_pred')
    axs[0].grid()
    axs[1].grid()
    
    plt.figure('Q_test vs. Q_pred')                                               
    fig, axs = plt.subplots(2)
    axs[0].plot(f, test_results['Qa'][case], 'r')
    axs[0].plot(f, pred_results['Qa'][case], 'b')       
    axs[0].set(ylabel='Qa')
    axs[0].legend(('Qa_test', 'Qa_pred'))
    axs[1].plot(f, test_results['Qb'][case], 'r')
    axs[1].plot(f, pred_results['Qb'][case], 'b')  
    axs[1].set(ylabel='Qb')   
    axs[1].set(xlabel='Frequency (GHz)')   
    if pred_eq_ckt_results != None:
        axs[0].plot(f, pred_eq_ckt_results['Qa'][case], 'g')
        axs[1].plot(f, pred_eq_ckt_results['Qb'][case], 'g')
        axs[0].legend(('Qa_test', 'Qa_pred', 'Qa_pred_eq_ckt'))
        axs[1].legend(('Qb_test', 'Qb_pred', 'Qb_pred_eq_ckt'))
        fig.suptitle('Q_test vs. Q_pred vs. R_pred_eq_ckt')
    else:
        axs[0].legend(('Qa_test', 'Qa_pred'))
        axs[1].legend(('Qb_test', 'Qb_pred'))                     
        fig.suptitle('Q_test vs. Q_pred')
    axs[0].grid()
    axs[1].grid()
    
    
    plt.figure('R_test vs. R_pred')                                               
    fig, axs = plt.subplots(2)
    axs[0].plot(f, test_results['Ra'][case], 'r')
    axs[0].plot(f, pred_results['Ra'][case], 'b')       
    axs[0].set(ylabel='Ra ($\Omega$)')
    axs[0].legend(('Ra_test', 'Ra_pred'))
    axs[1].plot(f, test_results['Rb'][case], 'r')
    axs[1].plot(f, pred_results['Rb'][case], 'b')  
    axs[1].set(ylabel='Rb ($\Omega$)')   
    axs[1].set(xlabel='Frequency (GHz)')   
    if pred_eq_ckt_results != None:
        axs[0].plot(f, pred_eq_ckt_results['Ra'][case], 'g')
        axs[1].plot(f, pred_eq_ckt_results['Rb'][case], 'g')
        axs[0].legend(('Ra_test', 'Ra_pred', 'Ra_pred_eq_ckt'))
        axs[1].legend(('Rb_test', 'Rb_pred', 'Rb_pred_eq_ckt'))
        fig.suptitle('R_test vs. R_pred vs. R_pred_eq_ckt')
    else:
        axs[0].legend(('Ra_test', 'Ra_pred'))
        axs[1].legend(('Rb_test', 'Rb_pred'))                     
        fig.suptitle('R_test vs. R_pred')
    axs[0].grid()
    axs[1].grid()
    
    plt.figure('k_test vs. k_pred')                                               
    plt.plot(f, test_results['k'][case], 'r')
    plt.plot(f, pred_results['k'][case], 'b')
    if pred_eq_ckt_results != None:
        plt.plot(f, pred_eq_ckt_results['k'][case], 'g')
        plt.legend(('k_test', 'k_pred', 'k_pred_eq_ckt'))
        plt.title('k_test vs. k_pred vs. k_pred_eq_ckt')
    else:
        plt.legend(('k_test', 'k_pred'))
        plt.title('k_test vs. k_pred')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('k')
    plt.grid()

    plt.figure('SRF_test vs. SRF_pred')                                               
    fig, axs = plt.subplots(2)
    if pred_eq_ckt_results == None:
        axs[0].plot(np.array(test_results['fr'])/1e9, np.array(pred_results['fr'])/1e9, '^')
        axs[0].set(ylabel='Predicted SRF (GHz)')
        axs[0].set(xlabel='True SRF (GHz)')   
        axs[1].hist(np.abs(np.array(test_results['fr'])/1e9 - np.array(pred_results['fr'])/1e9))
        axs[1].set(ylabel='Count')   
        axs[1].set(xlabel='SRF Prediction Error (GHz)')   
        fig.suptitle('SRF_test vs. SRF_pred')
        axs[0].grid()
        axs[1].grid()
        plt.tight_layout()
    else:
        axs[0].plot(np.array(test_results['fr'])/1e9, np.array(pred_eq_ckt_results['fr'])/1e9, '^')
        axs[0].set(ylabel='Predicted SRF (GHz)')
        axs[0].set(xlabel='True SRF (GHz)')   
        axs[1].hist(np.abs(np.array(test_results['fr'])/1e9 - np.array(pred_eq_ckt_results['fr'])/1e9))
        axs[1].set(ylabel='Count')   
        axs[1].set(xlabel='SRF Prediction Error (GHz)')   
        fig.suptitle('SRF_test vs. SRF_pred_eq_ckt')
        axs[0].grid()
        axs[1].grid()
        plt.tight_layout()

def rmse_plot():
    ## RMSE plot ##
    f = np.arange(0, freq_size_sliced)/1

    s11_real_rmse = np.sqrt(np.mean(abs(np.real(test_results['s_params'][:,:,0,0])-np.real(pred_results['s_params'][:,:,0,0]))**2, axis=0))
    s12_real_rmse = np.sqrt(np.mean(abs(np.real(test_results['s_params'][:,:,0,1])-np.real(pred_results['s_params'][:,:,0,1]))**2, axis=0))
    s13_real_rmse = np.sqrt(np.mean(abs(np.real(test_results['s_params'][:,:,0,2])-np.real(pred_results['s_params'][:,:,0,2]))**2, axis=0))
    s22_real_rmse = np.sqrt(np.mean(abs(np.real(test_results['s_params'][:,:,1,1])-np.real(pred_results['s_params'][:,:,1,1]))**2, axis=0))
    s23_real_rmse = np.sqrt(np.mean(abs(np.real(test_results['s_params'][:,:,2,1])-np.real(pred_results['s_params'][:,:,2,1]))**2, axis=0))
    s33_real_rmse = np.sqrt(np.mean(abs(np.real(test_results['s_params'][:,:,2,2])-np.real(pred_results['s_params'][:,:,2,2]))**2, axis=0))
    
    s11_imag_rmse = np.sqrt(np.mean(abs(np.imag(test_results['s_params'][:,:,0,0])-np.imag(pred_results['s_params'][:,:,0,0]))**2, axis=0))
    s12_imag_rmse = np.sqrt(np.mean(abs(np.imag(test_results['s_params'][:,:,0,1])-np.imag(pred_results['s_params'][:,:,0,1]))**2, axis=0))
    s13_imag_rmse = np.sqrt(np.mean(abs(np.imag(test_results['s_params'][:,:,0,2])-np.imag(pred_results['s_params'][:,:,0,2]))**2, axis=0))
    s22_imag_rmse = np.sqrt(np.mean(abs(np.imag(test_results['s_params'][:,:,1,1])-np.imag(pred_results['s_params'][:,:,1,1]))**2, axis=0))
    s23_imag_rmse = np.sqrt(np.mean(abs(np.imag(test_results['s_params'][:,:,2,1])-np.imag(pred_results['s_params'][:,:,2,1]))**2, axis=0))
    s33_imag_rmse = np.sqrt(np.mean(abs(np.imag(test_results['s_params'][:,:,2,2])-np.imag(pred_results['s_params'][:,:,2,2]))**2, axis=0))
    
    s11_mean_evm = np.mean(np.sqrt((np.real(test_results['s_params'][:,:,0,0])-np.real(pred_results['s_params'][:,:,0,0]))**2 + (np.imag(test_results['s_params'][:,:,0,0])-np.imag(pred_results['s_params'][:,:,0,0]))**2), axis=0)
    s12_mean_evm = np.mean(np.sqrt((np.real(test_results['s_params'][:,:,0,1])-np.real(pred_results['s_params'][:,:,0,1]))**2 + (np.imag(test_results['s_params'][:,:,0,1])-np.imag(pred_results['s_params'][:,:,0,1]))**2), axis=0)
    s13_mean_evm = np.mean(np.sqrt((np.real(test_results['s_params'][:,:,0,2])-np.real(pred_results['s_params'][:,:,0,2]))**2 + (np.imag(test_results['s_params'][:,:,0,2])-np.imag(pred_results['s_params'][:,:,0,2]))**2), axis=0)
    s22_mean_evm = np.mean(np.sqrt((np.real(test_results['s_params'][:,:,1,1])-np.real(pred_results['s_params'][:,:,1,1]))**2 + (np.imag(test_results['s_params'][:,:,1,1])-np.imag(pred_results['s_params'][:,:,1,1]))**2), axis=0)
    s23_mean_evm = np.mean(np.sqrt((np.real(test_results['s_params'][:,:,2,1])-np.real(pred_results['s_params'][:,:,2,1]))**2 + (np.imag(test_results['s_params'][:,:,2,1])-np.imag(pred_results['s_params'][:,:,2,1]))**2), axis=0)
    s33_mean_evm = np.mean(np.sqrt((np.real(test_results['s_params'][:,:,2,2])-np.real(pred_results['s_params'][:,:,2,2]))**2 + (np.imag(test_results['s_params'][:,:,2,2])-np.imag(pred_results['s_params'][:,:,2,2]))**2), axis=0)
 
    plt.figure('s11 mean evm')                                               
    plt.plot(f, s11_mean_evm, 'b')
    plt.xlabel('Frequency (GHz)', fontweight='bold')
    plt.ylabel('$\mathbf{S_{11}}$ EVM Mean', fontweight='bold')
    plt.grid(linewidth=2)
    plt.savefig(f'/autofs/fs1.ece/fs1.eecg.tcc/lizongh2/S-TCNN/plot/evm/EVM_Mean_S11_256GHz.eps',format='eps', bbox_inches='tight')


    plt.figure('s12 mean evm')                                               
    plt.plot(f, s12_mean_evm, 'b')
    plt.xlabel('Frequency (GHz)',fontweight='bold')
    plt.ylabel('$\mathbf{S_{12}}$ EVM Mean',fontweight='bold')
    plt.grid(linewidth=2)
    plt.savefig(f'/autofs/fs1.ece/fs1.eecg.tcc/lizongh2/S-TCNN/plot/evm/EVM_Mean_S12_256GHz.eps',format='eps', bbox_inches='tight')

    
    plt.figure('s13 mean evm')                                               
    plt.plot(f, s13_mean_evm, 'b')
    plt.xlabel('Frequency (GHz)',fontweight='bold')
    plt.ylabel('$\mathbf{S_{13}}$ EVM Mean',fontweight='bold')
    plt.grid(linewidth=2)
    plt.savefig(f'/autofs/fs1.ece/fs1.eecg.tcc/lizongh2/S-TCNN/plot/evm/EVM_Mean_S13_256GHz.eps',format='eps', bbox_inches='tight')

    
    plt.figure('s22 mean evm')                                               
    plt.plot(f, s22_mean_evm, 'b')
    plt.xlabel('Frequency (GHz)',fontweight='bold')
    plt.ylabel('$\mathbf{S_{22}}$ EVM Mean',fontweight='bold')
    plt.grid(linewidth=2)
    plt.savefig(f'/autofs/fs1.ece/fs1.eecg.tcc/lizongh2/S-TCNN/plot/evm/EVM_Mean_S22_256GHz.eps',format='eps', bbox_inches='tight')
    
    plt.figure('s23 mean evm')                                               
    plt.plot(f, s23_mean_evm, 'b')
    plt.xlabel('Frequency (GHz)',fontweight='bold')
    plt.ylabel('$\mathbf{S_{23}}$ EVM Mean',fontweight='bold')
    plt.grid(linewidth=2)
    plt.savefig(f'/autofs/fs1.ece/fs1.eecg.tcc/lizongh2/S-TCNN/plot/evm/EVM_Mean_S23_256GHz.eps',format='eps', bbox_inches='tight')

    
    plt.figure('s33 mean evm')                                               
    plt.plot(f, s33_mean_evm, 'b')
    plt.xlabel('Frequency (GHz)',fontweight='bold')
    plt.ylabel('$\mathbf{S_{33}}$ EVM Mean',fontweight='bold')
    plt.grid(linewidth=2)    
    plt.savefig(f'/autofs/fs1.ece/fs1.eecg.tcc/lizongh2/S-TCNN/plot/evm/EVM_Mean_S33_256GHz.eps',format='eps', bbox_inches='tight')

    # this picture shows all S11 - S33 mean evm in one plot
    plt.figure('S-parameters mean evm')
    plt.plot(f, s11_mean_evm, '-r')
    plt.plot(f, s12_mean_evm, '-b')
    plt.plot(f, s13_mean_evm, '-g')
    plt.plot(f, s22_mean_evm, '-.r')
    plt.plot(f, s23_mean_evm, '-.b')
    plt.plot(f, s33_mean_evm, '-.g')
    plt.xlabel('Frequency (GHz)', fontweight = 'bold')
    plt.ylabel('EVM Mean')
    plt.legend(['$\mathbf{S_{11}}$', '$\mathbf{S_{12}}$', '$\mathbf{S_{13}}$', '$\mathbf{S_{22}}$', '$\mathbf{S_{23}}$', '$\mathbf{S_{33}}$'])
    plt.grid(linewidth=2)
    plt.savefig(f'/autofs/fs1.ece/fs1.eecg.tcc/lizongh2/S-TCNN/plot/evm/EVM_Mean_256GHz.eps', format='eps', bbox_inches='tight')
    
    plt.figure('s11_test vs. s11_pred RMSE')
    fig, ax1 = plt.subplots()
    fig.suptitle('s11_test vs. s11_pred RMSE')
    ax2 = ax1.twinx()
    ax1.plot(f, s11_real_rmse, 'b-', label='real(s11)_rmse')
    ax2.plot(f, s11_imag_rmse, 'r-', label='imag(s11)_rmse')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Real_rmse', color='b')
    ax2.set_ylabel('Imag_rmse', color='r')
    fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    ax1.grid()
    plt.show()
    
    plt.figure('s12_test vs. s12_pred RMSE')
    fig, ax1 = plt.subplots()
    fig.suptitle('s12_test vs. s12_pred RMSE')
    ax2 = ax1.twinx()
    ax1.plot(f, s12_real_rmse, 'b-', label='real(s12)_rmse')
    ax2.plot(f, s12_imag_rmse, 'r-', label='imag(s12)_rmse')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Real_rmse', color='b')
    ax2.set_ylabel('Imag_rmse', color='r')
    fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    ax1.grid()
    plt.show()
    
    plt.figure('s13_test vs. s13_pred RMSE')
    fig, ax1 = plt.subplots()
    fig.suptitle('s13_test vs. s13_pred RMSE')
    ax2 = ax1.twinx()
    ax1.plot(f, s13_real_rmse, 'b-', label='real(s13)_rmse')
    ax2.plot(f, s13_imag_rmse, 'r-', label='imag(s13)_rmse')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Real_rmse', color='b')
    ax2.set_ylabel('Imag_rmse', color='r')
    fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    ax1.grid()
    plt.show()

    plt.figure('s22_test vs. s22_pred RMSE')
    fig, ax1 = plt.subplots()
    fig.suptitle('s22_test vs. s22_pred RMSE')
    ax2 = ax1.twinx()
    ax1.plot(f, s22_real_rmse, 'b-', label='real(s22)_rmse')
    ax2.plot(f, s22_imag_rmse, 'r-', label='imag(s22)_rmse')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Real_rmse', color='b')
    ax2.set_ylabel('Imag_rmse', color='r')
    fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    ax1.grid()
    plt.show()

    plt.figure('s23_test vs. s23_pred RMSE')
    fig, ax1 = plt.subplots()
    fig.suptitle('s23_test vs. s23_pred RMSE')
    ax2 = ax1.twinx()
    ax1.plot(f, s23_real_rmse, 'b-', label='real(s23)_rmse')
    ax2.plot(f, s23_imag_rmse, 'r-', label='imag(s23)_rmse')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Real_rmse', color='b')
    ax2.set_ylabel('Imag_rmse', color='r')
    fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    ax1.grid()
    plt.show()

    plt.figure('s33_test vs. s33_pred RMSE')
    fig, ax1 = plt.subplots()
    fig.suptitle('s33_test vs. s33_pred RMSE')
    ax2 = ax1.twinx()
    ax1.plot(f, s33_real_rmse, 'b-', label='real(s33)_rmse')
    ax2.plot(f, s33_imag_rmse, 'r-', label='imag(s33)_rmse')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Real_rmse', color='b')
    ax2.set_ylabel('Imag_rmse', color='r')
    fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    ax1.grid()
    plt.show()

plot(test_results, pred_results, case=1)
rmse_plot()


