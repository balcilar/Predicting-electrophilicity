import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error, r2_score

def loadCobraData(fname='cobradat.mat'):

    # read mat file
    mat = loadmat(fname)
    # make Adjagency and Connectivity matrixes as list
    A=[];C=[]
    for i in range(0,mat['A'].shape[0]):
        A.append(mat['A'][i][0])
        C.append(mat['C'][i][0])
    # read global features descriptors
    F=mat['F']

    # read global descr names
    Vnames=[]
    for i in range(0,mat['Vnames'][0].shape[0]):
        Vnames.append(mat['Vnames'][0][i][0])

    # read file name and molecule names
    FILE=[];NAME=[]
    for i in range(0,mat['FILE'].shape[0]):
        FILE.append(mat['FILE'][i][0][0])
        NAME.append(mat['NAME'][i][0][0])

    # read atomic descriptor name
    Anames=[]
    for i in range(0,mat['Anames'].shape[1]):
        Anames.append(mat['Anames'][0][i][0])
    # read atomic descriptors
    TT=[];Atom=[]
    for i in range(0,mat['TT'].shape[0]):
        TT.append(mat['TT'][i][0])
        SA=[]
        for j in range(0,mat['Atom'][i][0].shape[0]):
            SA.append(mat['Atom'][i][0][j][0][0])
        Atom.append(SA)
    #TT Atom Anames 

    return A,C,F,TT,Atom,Anames,Vnames,FILE,NAME

# read dataset
A,C,F,TT,Atom,Anames,Vnames,FILE,NAME=loadCobraData(fname='cobradat.mat')

Y=F[:,-1]


fname="outputs/gbdt_10fold_maxatm_glob_topology_predictions.csv"
#fname='outputs/RF_10fold_maxatm_glob_topology.csv'

df = pd.read_csv(fname) 


X = df.iloc[:,1:]

D=[]
for i in range(0,X.shape[1]):
    D.append(Y-X.iloc[:, i])
D=np.array(D).T


a=[];b=[]
for i in range(X.shape[1]):
    a.append(r2_score(Y,np.array(X.iloc[:,i])))
    b.append(np.mean(np.abs(Y-np.array(X.iloc[:,i]))))  
    #a.append(r2_score(Y,Y-np.array(X.iloc[:,i])))
    #b.append(np.mean(np.abs(np.array(X.iloc[:,i])))) 
    

txt1='MAE = '+str(np.fix(np.mean(b)*10000)/10000)+' ± '+str(np.fix(1000*np.std(b))/1000)
txt2='R2 = '+str(np.fix(np.mean(a)*10000)/10000)+' ± '+str(np.fix(1000*np.std(a))/1000)
txt3='STErr = ' +  str(np.fix(np.mean(np.std(D,axis=0))*1000)/1000) +' ± '+  str(np.fix(np.std(np.std(D,axis=0))*1000)/1000)

txt1='MAE = '+str(np.fix(np.mean(b)*10000)/10000)
txt2='R2 = '+str(np.fix(np.mean(a)*10000)/10000)


plt.figure(figsize=(10, 10))
ind = np.arange(111) 
plt.bar(ind,np.mean(D,axis=1),yerr=np.std(D,axis=1))
#plt.xlabel('Sample ID', fontsize=12)
#plt.ylabel('Prediction Error', fontsize=12)
#plt.text(40, 4, txt1+' '+txt2)
plt.show()

a=np.abs(np.mean(np.abs(D),axis=1)).mean()

if True:
    plt.figure(figsize=(10, 10))
    for i in range(X.shape[1]):
        plt.plot(Y,X.iloc[:, i],  "bo", alpha=1. / X.shape[1])    
    plt.plot(Y,np.mean(np.array(X),axis=1),  "r.")
    plt.plot([Y.min(), Y.max()],[Y.min(), Y.max()], "g--")
    plt.grid()
    plt.xlim(Y.min()-4,Y.max()+4)
    plt.ylim(Y.min()-4,Y.max()+4)
    # plt.xlabel('Actual electrophilicity', fontsize=12)
    # plt.ylabel('Predicted electrophilicity', fontsize=12)    
    # plt.text(-10, 10, txt1+' '+txt2)
    plt.show()





