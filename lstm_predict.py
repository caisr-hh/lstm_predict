# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 08:51:35 2019

@author: aaq109
For queries Contact: awais.ashfaq@hh.se
"""
import timeit
import numpy as np
from numpy import array
from keras.datasets import imdb
from keras.models import *
from keras.layers import *
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

# Read data
#Set Params
N_visits=15 # Maximum number of inpatient visits in the dataset

def read_data(exp, N_visits):
    label='sampledata_lstm_'+str(N_visits)+'.csv' # Dummy dataset for arranging the input data.
    print('Reading File: ',label)
    pidAdmMap = {}
    admDetailMap={}
    output=[]
    Weights=[]
    VisitIds=[]
    if exp[0:2]=='11': # Features used for different experiments according to the manuscript
        ind1=6
        ind2=202       
    elif exp[0:2]=='10':
        ind1=6
        ind2=17
    else:
        ind1=17
        ind2=202
    infd = open (label,'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        pid=int(tokens[0])
        admId=(tokens[1])
        det=(tokens[ind1:ind2]) #200 if 185 d2v vector is used
        output.append(tokens[5])      
        Weights.append(tokens[203]) 
        VisitIds.append(tokens[1])
        if admId in admDetailMap:
            admDetailMap[admId].append(det)
        else:
            admDetailMap[admId]=det
        if pid in pidAdmMap:
            pidAdmMap[pid].append(admId)
        else:
            pidAdmMap[pid]=[admId]
    infd.close()   
    _list = []
    for patient in pidAdmMap.keys():
        a = [admDetailMap[xx] for xx in pidAdmMap[patient]]
        _list.append(a)    
    X=np.array([np.array(xi) for xi in _list])   
    a,b,c=X.shape
    Y=np.array(output)
    Sample_weight=np.array(Weights)
    X = X.astype(np.float)
    Y = Y.astype(np.float)
    Sample_weight = Sample_weight.astype(np.float)
    Y=Y.reshape(X.shape[0],N_visits,1)
    return X, Y,Sample_weight,VisitIds
    

def ppv(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    ppv = true_positives / (predicted_positives + K.epsilon())
    return ppv
    
def npv(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    predicted_negatives = K.sum(K.round(K.clip(1-y_pred, 0, 1)))
    npv = true_negatives / (predicted_negatives + K.epsilon())
    return npv
    
    
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())
    
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())
    
def threshold_binary_accuracy(y_true, y_pred):
    threshold = 0.1
    if K.backend() == 'tensorflow':
        return K.mean(K.equal(y_true, K.tf.cast(K.less(y_pred,threshold), y_true.dtype)))
    else:
        return K.mean(K.equal(y_true, K.less(y_pred,threshold)))
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def model_eval(model, X_test,Y_test, Sample_weight_test, thresh,exp):
    
    y_pred = model.predict(X_test).ravel()
    y_test=Y_test.ravel()
    g=Sample_weight_test.ravel()
    g[g==W_classC]=1
    g[g==W_classB]=1
    g[g==W_classA]=0
    if exp[2]=='1':
        fpr, tpr, thetas = roc_curve(y_test, y_pred,sample_weight=g)
        prc, recal, thetas = precision_recall_curve(y_test, y_pred,sample_weight=g)
        indices=np.where(g==0) #Patient gender
        y_pred=np.delete(y_pred,indices,0)
        y_test=np.delete(y_test,indices,0)
    else:
        fpr, tpr, thetas = roc_curve(y_test, y_pred)
        prc, recal, thetas = precision_recall_curve(y_test, y_pred)
        
    auc_30d = auc(fpr, tpr)
    pr_auc = auc(recal,prc)
    ##print('ROC-AUC : ', auc_30d, 'for iteration ', iter_nm )
    
    y_pred[y_pred>=thresh]=1
    y_pred[y_pred<thresh]=0
    cm= confusion_matrix(y_test, y_pred)
    #print('Confusion Matrix : \n', cm, 'for iteration ', iter_nm )
    Accuracy=(cm[0,0]+cm[1,1])/sum(sum(cm))   
    AUC_test=auc_30d
    Sensitivity_test=cm[1,1]/(cm[1,0]+cm[1,1])
    Specificity_test=cm[0,0]/(cm[0,0]+cm[0,1])
    PR_auc=pr_auc
    return Accuracy, AUC_test, Sensitivity_test, Specificity_test, PR_auc

def save_print(AUC_test, Sensitivity_test, Specificity_test, PR_auc, exp):
    label1='AUC_test_'+exp+'.npy'
    label2='Sensitivity_test_'+exp+'.npy'
    label3='Specificity_test_'+exp+'.npy'
    label4='PR_auc_'+exp+'.npy'
    np.save(label1, AUC_test) 
    np.save(label2, Sensitivity_test) 
    np.save(label3, Specificity_test) 
    np.save(label4, PR_auc) 
    val1=np.fromiter(AUC_test.values(), dtype=float)
    val2=np.fromiter(Sensitivity_test.values(), dtype=float)
    val3=np.fromiter(Specificity_test.values(), dtype=float)
    val4=np.fromiter(PR_auc.values(), dtype=float)
    print(label1,[np.mean(val1),np.std(val1)])
    print(label2,[np.mean(val2),np.std(val2)])
    print(label3,[np.mean(val3),np.std(val3)])                                                                                                                                                                                                                                                                                                                                                               
    print(label4,[np.mean(val4),np.std(val4)])
    return None
    
## Define different experiments
    # 1111 - HDF+MDF+LSTM+CA
exp='1111'
AUC_test={}
Accuracy_test={}
PR_auc={}
Sensitivity_test={}
Specificity_test={}
average_precision={}
#Set Params
W_classA=0 #Dummy visit weights
W_classB=1 #No readmission class weight
W_classC=3 #Readmission class weight
E_pochs=100 # Traning epochs - 80 is better based on ES result
B_size=32 # Batch size
T_size=0.3 # Samples used for testing
NN_nodes=[128,64,32,1] # Number of nodes in the NN
N_iter=10
thresh=0.5 # Discrimination threshold
#label='rnn_new_comorb_2016_'+str(N_visits)+'.csv'
X, Y, Sample_weight,VisitIds=read_data(exp, N_visits)
Sample_weight[Sample_weight==0]=W_classA
Sample_weight[Sample_weight==1]=W_classB
Sample_weight[Sample_weight==2]=W_classC
Sample_weight=Sample_weight.reshape(X.shape[0],N_visits,1)
Visits=np.array(VisitIds)
Visits=Visits.reshape(X.shape[0],N_visits,1)

for iter_nm in range(0,N_iter):
    print('Iteration ',iter_nm)    
    X_train, X_test, Y_train, Y_test, Sample_weight_train, Sample_weight_test, Visit_train, Visit_test = train_test_split(X, Y,Sample_weight,Visits, test_size=T_size, shuffle=True)
    Sample_weight_train=Sample_weight_train.reshape(len(Sample_weight_train),N_visits)
    model = Sequential()  
    model.add(TimeDistributed(Dense(NN_nodes[0], activation='sigmoid'), input_shape=(N_visits, X.shape[2])))
    model.add(TimeDistributed(Dense(NN_nodes[1], activation='sigmoid'))) 
    model.add(LSTM(NN_nodes[2], return_sequences=True))
    model.add(TimeDistributed(Dense(NN_nodes[3], activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',sample_weight_mode='temporal', metrics=[sensitivity, specificity, ppv, npv, 'accuracy'])
    print(model.summary())
    #np.random.seed(1337)
    print('Training start', 'for iteration ', iter_nm ) 
    model.fit(X_train, Y_train, epochs=E_pochs, batch_size=B_size, verbose=0, sample_weight=Sample_weight_train,shuffle=True)      
    print('Training complete', 'for iteration ', iter_nm ) 
    print('Evaluation', 'for iteration ', iter_nm )    
    Accuracy_test[iter_nm], AUC_test[iter_nm], Sensitivity_test[iter_nm], Specificity_test[iter_nm], PR_auc[iter_nm]=model_eval(model, X_test,Y_test, Sample_weight_test, thresh,exp)
    print('Evaluation complete', 'for iteration ', iter_nm ) 

save_print(AUC_test, Sensitivity_test, Specificity_test, PR_auc, exp)


## Define different experiments
    # 1110 - HDF+MDF+LSTM
exp='1110'
AUC_test={}
Accuracy_test={}
PR_auc={}
Sensitivity_test={}
Specificity_test={}
average_precision={}
#Set Params
W_classA=0 #Dummy visit weights
W_classB=1 #No readmission class weight
W_classC=1 #Readmission class weight
E_pochs=100 # Traning epochs - 80 is better based on ES result
B_size=32 # Batch size
T_size=0.3 # Samples used for testing
NN_nodes=[128,64,32,1] # Number of nodes in the NN
N_iter=10
thresh=0.5 # Discrimination threshold
#label='rnn_new_comorb_2016_'+str(N_visits)+'.csv'
X, Y, Sample_weight,VisitIds=read_data(exp, N_visits)
Sample_weight[Sample_weight==0]=W_classA
Sample_weight[Sample_weight==1]=W_classB
Sample_weight[Sample_weight==2]=W_classC
Sample_weight=Sample_weight.reshape(X.shape[0],N_visits,1)
Visits=np.array(VisitIds)
Visits=Visits.reshape(X.shape[0],N_visits,1)

for iter_nm in range(0,N_iter):
    print('Iteration ',iter_nm)    
    X_train, X_test, Y_train, Y_test, Sample_weight_train, Sample_weight_test, Visit_train, Visit_test = train_test_split(X, Y,Sample_weight,Visits, test_size=T_size, shuffle=True)
    Sample_weight_train=Sample_weight_train.reshape(len(Sample_weight_train),N_visits)
    model = Sequential()  
    model.add(TimeDistributed(Dense(NN_nodes[0], activation='sigmoid'), input_shape=(N_visits, X.shape[2])))
    model.add(TimeDistributed(Dense(NN_nodes[1], activation='sigmoid'))) 
    model.add(LSTM(NN_nodes[2], return_sequences=True))
    model.add(TimeDistributed(Dense(NN_nodes[3], activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',sample_weight_mode='temporal', metrics=[sensitivity, specificity, ppv, npv, 'accuracy'])
    print(model.summary())
    #np.random.seed(1337)
    print('Training start', 'for iteration ', iter_nm ) 
    model.fit(X_train, Y_train, epochs=E_pochs, batch_size=B_size, verbose=0, sample_weight=Sample_weight_train,shuffle=True)      
    print('Training complete', 'for iteration ', iter_nm )
    print('Evaluation', 'for iteration ', iter_nm )    
    Accuracy_test[iter_nm], AUC_test[iter_nm], Sensitivity_test[iter_nm], Specificity_test[iter_nm], PR_auc[iter_nm]=model_eval(model, X_test,Y_test, Sample_weight_test, thresh,exp)
    print('Evaluation complete', 'for iteration ', iter_nm ) 

save_print(AUC_test, Sensitivity_test, Specificity_test, PR_auc, exp)


## Define different experiments
    # 0111 - MDF+LSTM+CA
exp='0111'
AUC_test={}
Accuracy_test={}
PR_auc={}
Sensitivity_test={}
Specificity_test={}
average_precision={}
#Set Params
W_classA=0 #Dummy visit weights
W_classB=1 #No readmission class weight
W_classC=3 #Readmission class weight
E_pochs=100 # Traning epochs - 80 is better based on ES result
B_size=32 # Batch size
T_size=0.3 # Samples used for testing
NN_nodes=[128,64,32,1] # Number of nodes in the NN
N_iter=10
thresh=0.5 # Discrimination threshold
#label='rnn_new_comorb_2016_'+str(N_visits)+'.csv'
X, Y, Sample_weight,VisitIds=read_data(exp, N_visits)
Sample_weight[Sample_weight==0]=W_classA
Sample_weight[Sample_weight==1]=W_classB
Sample_weight[Sample_weight==2]=W_classC
Sample_weight=Sample_weight.reshape(X.shape[0],N_visits,1)
Visits=np.array(VisitIds)
Visits=Visits.reshape(X.shape[0],N_visits,1)

for iter_nm in range(0,N_iter):
    print('Iteration ',iter_nm)    
    X_train, X_test, Y_train, Y_test, Sample_weight_train, Sample_weight_test, Visit_train, Visit_test = train_test_split(X, Y,Sample_weight,Visits, test_size=T_size, shuffle=True)
    Sample_weight_train=Sample_weight_train.reshape(len(Sample_weight_train),N_visits)
    model = Sequential()  
    model.add(TimeDistributed(Dense(NN_nodes[0], activation='sigmoid'), input_shape=(N_visits, X.shape[2])))
    model.add(TimeDistributed(Dense(NN_nodes[1], activation='sigmoid'))) 
    model.add(LSTM(NN_nodes[2], return_sequences=True))
    model.add(TimeDistributed(Dense(NN_nodes[3], activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',sample_weight_mode='temporal', metrics=[sensitivity, specificity, ppv, npv, 'accuracy'])
    print(model.summary())
    #np.random.seed(1337)
    print('Training start', 'for iteration ', iter_nm ) 
    model.fit(X_train, Y_train, epochs=E_pochs, batch_size=B_size, verbose=0, sample_weight=Sample_weight_train,shuffle=True)      
    print('Training complete', 'for iteration ', iter_nm ) 
    print('Evaluation', 'for iteration ', iter_nm )    
    Accuracy_test[iter_nm], AUC_test[iter_nm], Sensitivity_test[iter_nm], Specificity_test[iter_nm], PR_auc[iter_nm]=model_eval(model, X_test,Y_test, Sample_weight_test, thresh,exp)
    print('Evaluation complete', 'for iteration ', iter_nm ) 

save_print(AUC_test, Sensitivity_test, Specificity_test, PR_auc, exp)

## Define different experiments
    # 0110 - MDF+LSTM
exp='0110'
AUC_test={}
Accuracy_test={}
PR_auc={}
Sensitivity_test={}
Specificity_test={}
average_precision={}
#Set Params
W_classA=0 #Dummy visit weights
W_classB=1 #No readmission class weight
W_classC=1 #Readmission class weight
E_pochs=100 # Traning epochs - 80 is better based on ES result
B_size=32 # Batch size
T_size=0.3 # Samples used for testing
NN_nodes=[128,64,32,1] # Number of nodes in the NN
N_iter=10
thresh=0.5 # Discrimination threshold
#label='rnn_new_comorb_2016_'+str(N_visits)+'.csv'
X, Y, Sample_weight,VisitIds=read_data(exp, N_visits)
Sample_weight[Sample_weight==0]=W_classA
Sample_weight[Sample_weight==1]=W_classB
Sample_weight[Sample_weight==2]=W_classC
Sample_weight=Sample_weight.reshape(X.shape[0],N_visits,1)
Visits=np.array(VisitIds)
Visits=Visits.reshape(X.shape[0],N_visits,1)

for iter_nm in range(0,N_iter):
    print('Iteration ',iter_nm)    
    X_train, X_test, Y_train, Y_test, Sample_weight_train, Sample_weight_test, Visit_train, Visit_test = train_test_split(X, Y,Sample_weight,Visits, test_size=T_size, shuffle=True)
    Sample_weight_train=Sample_weight_train.reshape(len(Sample_weight_train),N_visits)
    model = Sequential()  
    model.add(TimeDistributed(Dense(NN_nodes[0], activation='sigmoid'), input_shape=(N_visits, X.shape[2])))
    model.add(TimeDistributed(Dense(NN_nodes[1], activation='sigmoid'))) 
    model.add(LSTM(NN_nodes[2], return_sequences=True))
    model.add(TimeDistributed(Dense(NN_nodes[3], activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',sample_weight_mode='temporal', metrics=[sensitivity, specificity, ppv, npv, 'accuracy'])
    print(model.summary())
    #np.random.seed(1337)
    print('Training start', 'for iteration ', iter_nm ) 
    model.fit(X_train, Y_train, epochs=E_pochs, batch_size=B_size, verbose=0, sample_weight=Sample_weight_train,shuffle=True)      
    print('Training complete', 'for iteration ', iter_nm ) 
    print('Evaluation', 'for iteration ', iter_nm )    
    Accuracy_test[iter_nm], AUC_test[iter_nm], Sensitivity_test[iter_nm], Specificity_test[iter_nm], PR_auc[iter_nm]=model_eval(model, X_test,Y_test, Sample_weight_test, thresh,exp)
    print('Evaluation complete', 'for iteration ', iter_nm ) 

save_print(AUC_test, Sensitivity_test, Specificity_test, PR_auc, exp)


## Define different experiments
    # 1011 - HDF+LSTM+CA
exp='1011'
AUC_test={}
Accuracy_test={}
PR_auc={}
Sensitivity_test={}
Specificity_test={}
average_precision={}
#Set Params
W_classA=0 #Dummy visit weights
W_classB=1 #No readmission class weight
W_classC=3 #Readmission class weight
E_pochs=100 # Traning epochs - 80 is better based on ES result
B_size=32 # Batch size
T_size=0.3 # Samples used for testing
NN_nodes=[6,3,1] # Number of nodes in the NN
N_iter=10
thresh=0.5 # Discrimination threshold
#label='rnn_new_comorb_2016_'+str(N_visits)+'.csv'
X, Y, Sample_weight,VisitIds=read_data(exp, N_visits)
Sample_weight[Sample_weight==0]=W_classA
Sample_weight[Sample_weight==1]=W_classB
Sample_weight[Sample_weight==2]=W_classC
Sample_weight=Sample_weight.reshape(X.shape[0],N_visits,1)
Visits=np.array(VisitIds)
Visits=Visits.reshape(X.shape[0],N_visits,1)

for iter_nm in range(0,N_iter):
    print('Iteration ',iter_nm)    
    X_train, X_test, Y_train, Y_test, Sample_weight_train, Sample_weight_test, Visit_train, Visit_test = train_test_split(X, Y,Sample_weight,Visits, test_size=T_size, shuffle=True)
    Sample_weight_train=Sample_weight_train.reshape(len(Sample_weight_train),N_visits)
    model = Sequential()  
    model.add(TimeDistributed(Dense(NN_nodes[0], activation='sigmoid'), input_shape=(N_visits, X.shape[2])))
   # model.add(TimeDistributed(Dense(NN_nodes[1], activation='sigmoid'))) 
    model.add(LSTM(NN_nodes[1], return_sequences=True))
    #model.add(TimeDistributed(Dense(NN_nodes[3], activation='sigmoid')))
    model.add(TimeDistributed(Dense(NN_nodes[2], activation='sigmoid')))   
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',sample_weight_mode='temporal', metrics=[sensitivity, specificity, ppv, npv, 'accuracy'])
    print(model.summary())
    #np.random.seed(1337)
    print('Training start', 'for iteration ', iter_nm ) 
    model.fit(X_train, Y_train, epochs=E_pochs, batch_size=B_size, verbose=0, sample_weight=Sample_weight_train,shuffle=True)      
    print('Training complete', 'for iteration ', iter_nm ) 
    print('Evaluation', 'for iteration ', iter_nm )    
    Accuracy_test[iter_nm], AUC_test[iter_nm], Sensitivity_test[iter_nm], Specificity_test[iter_nm], PR_auc[iter_nm]=model_eval(model, X_test,Y_test, Sample_weight_test, thresh,exp)
    print('Evaluation complete', 'for iteration ', iter_nm ) 

save_print(AUC_test, Sensitivity_test, Specificity_test, PR_auc, exp)


## Define different experiments
    # 1010 - HDF+LSTM
exp='1010'
AUC_test={}
Accuracy_test={}
PR_auc={}
Sensitivity_test={}
Specificity_test={}
average_precision={}
#Set Params
W_classA=0 #Dummy visit weights
W_classB=1 #No readmission class weight
W_classC=1 #Readmission class weight
E_pochs=100 # Traning epochs - 80 is better based on ES result
B_size=32 # Batch size
T_size=0.3 # Samples used for testing
NN_nodes=[6,3,1] # Number of nodes in the NN
N_iter=10
thresh=0.5 # Discrimination threshold
#label='rnn_new_comorb_2016_'+str(N_visits)+'.csv'
X, Y, Sample_weight,VisitIds=read_data(exp, N_visits)
Sample_weight[Sample_weight==0]=W_classA
Sample_weight[Sample_weight==1]=W_classB
Sample_weight[Sample_weight==2]=W_classC
Sample_weight=Sample_weight.reshape(X.shape[0],N_visits,1)
Visits=np.array(VisitIds)
Visits=Visits.reshape(X.shape[0],N_visits,1)

for iter_nm in range(0,N_iter):
    print('Iteration ',iter_nm)    
    X_train, X_test, Y_train, Y_test, Sample_weight_train, Sample_weight_test, Visit_train, Visit_test = train_test_split(X, Y,Sample_weight,Visits, test_size=T_size, shuffle=True)
    Sample_weight_train=Sample_weight_train.reshape(len(Sample_weight_train),N_visits)
    model = Sequential()  
    model.add(TimeDistributed(Dense(NN_nodes[0], activation='sigmoid'), input_shape=(N_visits, X.shape[2])))
   # model.add(TimeDistributed(Dense(NN_nodes[1], activation='sigmoid'))) 
    model.add(LSTM(NN_nodes[1], return_sequences=True))
    #model.add(TimeDistributed(Dense(NN_nodes[3], activation='sigmoid')))
    model.add(TimeDistributed(Dense(NN_nodes[2], activation='sigmoid')))   
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',sample_weight_mode='temporal', metrics=[sensitivity, specificity, ppv, npv, 'accuracy'])
    print(model.summary())
    #np.random.seed(1337)
    print('Training start', 'for iteration ', iter_nm ) 
    model.fit(X_train, Y_train, epochs=E_pochs, batch_size=B_size, verbose=0, sample_weight=Sample_weight_train,shuffle=True)      
    print('Training complete', 'for iteration ', iter_nm ) 
    print('Evaluation', 'for iteration ', iter_nm )    
    Accuracy_test[iter_nm], AUC_test[iter_nm], Sensitivity_test[iter_nm], Specificity_test[iter_nm], PR_auc[iter_nm]=model_eval(model, X_test,Y_test, Sample_weight_test, thresh,exp)
    print('Evaluation complete', 'for iteration ', iter_nm ) 

save_print(AUC_test, Sensitivity_test, Specificity_test, PR_auc, exp)


## Define different experiments
    # 1101 - HDF+MDF+CA
exp='1101'
AUC_test={}
Accuracy_test={}
PR_auc={}
Sensitivity_test={}
Specificity_test={}
average_precision={}
#Set Params
W_classA=0 #Dummy visit weights
W_classB=1 #No readmission class weight
W_classC=3 #Readmission class weight
E_pochs=100 # Traning epochs
B_size=32*N_visits # Batch size
T_size=0.3 # Samples used for testing
NN_nodes=[128,64,1] # Number of nodes in the NN
N_iter=10
thresh=0.5 # Discrimination threshold
#label='rnn_new_comorb_2016_'+str(N_visits)+'.csv'
X, Y, Sample_weight,VisitIds=read_data(exp, N_visits)
Sample_weight[Sample_weight==0]=W_classA
Sample_weight[Sample_weight==1]=W_classB
Sample_weight[Sample_weight==2]=W_classC
Visits=np.array(VisitIds)
a,b,c=X.shape
X=X.reshape(a*b,c)
Y=Y.reshape(a*b,1)
Sample_weight=Sample_weight.ravel()
Visits=Visits.reshape(a*N_visits,1)
ind=np.where(Sample_weight==0)
X=np.delete(X,ind,0)
Y=np.delete(Y,ind,0)
Sample_weight=np.delete(Sample_weight,ind,0)
Visits=np.delete(Visits,ind,0)
for iter_nm in range(0,N_iter):
    print('Iteration ',iter_nm)    
    X_train, X_test, Y_train, Y_test, Sample_weight_train, Sample_weight_test, Visit_train, Visit_test = train_test_split(X, Y,Sample_weight,Visits, test_size=T_size, shuffle=True)
    model = Sequential()   
    model.add(Dense(NN_nodes[0], activation='sigmoid', input_dim=c))  
    model.add(Dense(NN_nodes[1], activation='sigmoid'))            
    model.add(Dense(NN_nodes[2], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',sample_weight_mode='None', metrics=[sensitivity, specificity, ppv, npv, 'accuracy'])
    #model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary()) 
    model.fit(X_train, Y_train, epochs=E_pochs, batch_size=B_size, verbose=0, class_weight={0:W_classB,1:W_classC},shuffle=True)    
    print('Training complete', 'for iteration ', iter_nm ) 
    print('Evaluation', 'for iteration ', iter_nm )    
    Accuracy_test[iter_nm], AUC_test[iter_nm], Sensitivity_test[iter_nm], Specificity_test[iter_nm], PR_auc[iter_nm]=model_eval(model, X_test,Y_test, Sample_weight_test, thresh,exp)
    print('Evaluation complete', 'for iteration ', iter_nm ) 

save_print(AUC_test, Sensitivity_test, Specificity_test, PR_auc, exp)


## Define different experiments
    # 1101 - HDF+MDF
exp='1100'
AUC_test={}
Accuracy_test={}
PR_auc={}
Sensitivity_test={}
Specificity_test={}
average_precision={}
#Set Params
W_classA=0 #Dummy visit weights
W_classB=1 #No readmission class weight
W_classC=1 #Readmission class weight
E_pochs=100 # Traning epochs
B_size=32*N_visits # Batch size
T_size=0.3 # Samples used for testing
NN_nodes=[128,64,1] # Number of nodes in the NN
N_iter=10
thresh=0.5 # Discrimination threshold
#label='rnn_new_comorb_2016_'+str(N_visits)+'.csv'
X, Y, Sample_weight,VisitIds=read_data(exp, N_visits)
Sample_weight[Sample_weight==0]=W_classA
Sample_weight[Sample_weight==1]=W_classB
Sample_weight[Sample_weight==2]=W_classC
Visits=np.array(VisitIds)
a,b,c=X.shape
X=X.reshape(a*b,c)
Y=Y.reshape(a*b,1)
Sample_weight=Sample_weight.ravel()
Visits=Visits.reshape(a*N_visits,1)
ind=np.where(Sample_weight==0)
X=np.delete(X,ind,0)
Y=np.delete(Y,ind,0)
Sample_weight=np.delete(Sample_weight,ind,0)
Visits=np.delete(Visits,ind,0)
for iter_nm in range(0,N_iter):
    print('Iteration ',iter_nm)    
    X_train, X_test, Y_train, Y_test, Sample_weight_train, Sample_weight_test, Visit_train, Visit_test = train_test_split(X, Y,Sample_weight,Visits, test_size=T_size, shuffle=True)
    model = Sequential()   
    model.add(Dense(NN_nodes[0], activation='sigmoid', input_dim=c))  
    model.add(Dense(NN_nodes[1], activation='sigmoid'))            
    model.add(Dense(NN_nodes[2], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',sample_weight_mode='None', metrics=[sensitivity, specificity, ppv, npv, 'accuracy'])
    #model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary()) 
    model.fit(X_train, Y_train, epochs=E_pochs, batch_size=B_size, verbose=0, class_weight={0:W_classB,1:W_classC},shuffle=True)    
    print('Training complete', 'for iteration ', iter_nm ) 
    print('Evaluation', 'for iteration ', iter_nm )    
    Accuracy_test[iter_nm], AUC_test[iter_nm], Sensitivity_test[iter_nm], Specificity_test[iter_nm], PR_auc[iter_nm]=model_eval(model, X_test,Y_test, Sample_weight_test, thresh,exp)
    print('Evaluation complete', 'for iteration ', iter_nm ) 

save_print(AUC_test, Sensitivity_test, Specificity_test, PR_auc, exp)


## Define different experiments
    # 1001 - HDF+CA
exp='1001'
AUC_test={}
Accuracy_test={}
PR_auc={}
Sensitivity_test={}
Specificity_test={}
average_precision={}
#Set Params
W_classA=0 #Dummy visit weights
W_classB=1 #No readmission class weight
W_classC=3 #Readmission class weight
E_pochs=100 # Traning epochs
B_size=32*N_visits # Batch size
T_size=0.3 # Samples used for testing
NN_nodes=[128,64,1] # Number of nodes in the NN
N_iter=10
thresh=0.5 # Discrimination threshold
#label='rnn_new_comorb_2016_'+str(N_visits)+'.csv'
X, Y, Sample_weight,VisitIds=read_data(exp, N_visits)
Sample_weight[Sample_weight==0]=W_classA
Sample_weight[Sample_weight==1]=W_classB
Sample_weight[Sample_weight==2]=W_classC
Visits=np.array(VisitIds)
a,b,c=X.shape
X=X.reshape(a*b,c)
Y=Y.reshape(a*b,1)
Sample_weight=Sample_weight.ravel()
Visits=Visits.reshape(a*N_visits,1)
ind=np.where(Sample_weight==0)
X=np.delete(X,ind,0)
Y=np.delete(Y,ind,0)
Sample_weight=np.delete(Sample_weight,ind,0)
Visits=np.delete(Visits,ind,0)
for iter_nm in range(0,N_iter):
    print('Iteration ',iter_nm)    
    X_train, X_test, Y_train, Y_test, Sample_weight_train, Sample_weight_test, Visit_train, Visit_test = train_test_split(X, Y,Sample_weight,Visits, test_size=T_size, shuffle=True)
    model = Sequential()   
    model.add(Dense(NN_nodes[0], activation='sigmoid', input_dim=c))  
    model.add(Dense(NN_nodes[1], activation='sigmoid'))            
    model.add(Dense(NN_nodes[2], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',sample_weight_mode='None', metrics=[sensitivity, specificity, ppv, npv, 'accuracy'])
    #model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary()) 
    model.fit(X_train, Y_train, epochs=E_pochs, batch_size=B_size, verbose=0, class_weight={0:W_classB,1:W_classC},shuffle=True)    
    print('Training complete', 'for iteration ', iter_nm ) 
    print('Evaluation', 'for iteration ', iter_nm )    
    Accuracy_test[iter_nm], AUC_test[iter_nm], Sensitivity_test[iter_nm], Specificity_test[iter_nm], PR_auc[iter_nm]=model_eval(model, X_test,Y_test, Sample_weight_test, thresh,exp)
    print('Evaluation complete', 'for iteration ', iter_nm ) 

save_print(AUC_test, Sensitivity_test, Specificity_test, PR_auc, exp)


## Define different experiments
    # 1001 - HDF+CA
exp='1000'
AUC_test={}
Accuracy_test={}
PR_auc={}
Sensitivity_test={}
Specificity_test={}
average_precision={}
#Set Params
W_classA=0 #Dummy visit weights
W_classB=1 #No readmission class weight
W_classC=1 #Readmission class weight
E_pochs=100 # Traning epochs
B_size=32*N_visits # Batch size
T_size=0.3 # Samples used for testing
NN_nodes=[128,64,1] # Number of nodes in the NN
N_iter=10
thresh=0.5 # Discrimination threshold
#label='rnn_new_comorb_2016_'+str(N_visits)+'.csv'
X, Y, Sample_weight,VisitIds=read_data(exp, N_visits)
Sample_weight[Sample_weight==0]=W_classA
Sample_weight[Sample_weight==1]=W_classB
Sample_weight[Sample_weight==2]=W_classC
Visits=np.array(VisitIds)
a,b,c=X.shape
X=X.reshape(a*b,c)
Y=Y.reshape(a*b,1)
Sample_weight=Sample_weight.ravel()
Visits=Visits.reshape(a*N_visits,1)
ind=np.where(Sample_weight==0)
X=np.delete(X,ind,0)
Y=np.delete(Y,ind,0)
Sample_weight=np.delete(Sample_weight,ind,0)
Visits=np.delete(Visits,ind,0)
for iter_nm in range(0,N_iter):
    print('Iteration ',iter_nm)    
    X_train, X_test, Y_train, Y_test, Sample_weight_train, Sample_weight_test, Visit_train, Visit_test = train_test_split(X, Y,Sample_weight,Visits, test_size=T_size, shuffle=True)
    model = Sequential()   
    model.add(Dense(NN_nodes[0], activation='sigmoid', input_dim=c))  
    model.add(Dense(NN_nodes[1], activation='sigmoid'))            
    model.add(Dense(NN_nodes[2], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',sample_weight_mode='None', metrics=[sensitivity, specificity, ppv, npv, 'accuracy'])
    #model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary()) 
    model.fit(X_train, Y_train, epochs=E_pochs, batch_size=B_size, verbose=0, class_weight={0:W_classB,1:W_classC},shuffle=True)    
    print('Training complete', 'for iteration ', iter_nm ) 
    print('Evaluation', 'for iteration ', iter_nm )    
    Accuracy_test[iter_nm], AUC_test[iter_nm], Sensitivity_test[iter_nm], Specificity_test[iter_nm], PR_auc[iter_nm]=model_eval(model, X_test,Y_test, Sample_weight_test, thresh,exp)
    print('Evaluation complete', 'for iteration ', iter_nm ) 

save_print(AUC_test, Sensitivity_test, Specificity_test, PR_auc, exp)


#Significance
import numpy as np
AUC_1111=np.fromiter(np.load('AUC_test_1111.npy').item().values(), dtype=float)
AUC_1110=np.fromiter(np.load('AUC_test_1110.npy').item().values(), dtype=float)
AUC_1101=np.fromiter(np.load('AUC_test_1101.npy').item().values(), dtype=float)
AUC_1011=np.fromiter(np.load('AUC_test_1011.npy').item().values(), dtype=float)
AUC_0111=np.fromiter(np.load('AUC_test_0111.npy').item().values(), dtype=float)
from scipy import stats
a,b=stats.ttest_ind(AUC_1111,AUC_0111)
print(a,b)
