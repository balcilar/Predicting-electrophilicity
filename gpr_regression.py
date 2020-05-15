"""
@author: Muhammet Balcilar
LITIS Lab, Rouen, France
muhammetbalcilar@gmail.com


This script is the demostration of how we can export extracted Cobra Dataset.
You can write your own code by reading data from given dataset as well but we recommended you use our provided mat file

Dataset consist of 111 different molecules graph connections global and atomic descriptors.

Dataset consist of A,C,F,TT,Atom,Anames,Vnames,FILE,NAME variables. Here is their explanations.

A :     List of Agencency matrix; It consist of 111  variable size of binary valued matrix
C :     List of Connectivity matrix; It consist of 111 variable size of double valued matrix
F :     111x28 dimensional matrix keeps the global moleculer descriptor of each molecule
TT:     111 element of list. Each element is also matrix by number of atom of corresponding molecule row but 54 column
Atom:   111 element list. Each element also differnet length of list as well. Keeps the atom names. Forinstance Atom[0][0] shows theh name of the atom of 1st molecules 1st atom.
Anames: 54 length list ; keeps the name of atomic descriptor. Since we have 54 atomic descriptor it consist of 54 string
Vnames: 28 length list ; keeps the name of global descriptor. Since we have 28 global descriptor it consist of 28 string
FILE:   111 element of list. Keeps the file name of corresponding molecule
NAME:   111 element of list. Keeps the molecule name of corresponding molecule

"""


import numpy as np
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, WhiteKernel

import sklearn.gaussian_process as gp

import pandas as pd


# select feature 0:global  1:4stat-atomic  2: topologic 3:global+4stat atomic  4:global+4stat atomic+topologic
# 5: global+ reactive atomic + topologic 6: global+ max atomic+topologic  7:reactive atomic  8: reactive atomic +gglobal
feature=6
# if chemistry family code is included or not.
fmcode=1

# random seed
seed=42

# if you like to save shap significant
isshap=0

# number of runs
nsim=100



np.random.seed(seed)
if isshap==1:
    import shap
    shap.initjs()


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

# take electrophicity value from global feature table
Y=F[:,-1]
react=F[:,-2]

# remove extra added 11 gobal feature
F=F[:,0:16]

# remove extra added 4 atomic feature
for i in range(0,111):
    TT[i]=TT[i][:,4:]
    # remove problematic atomic descriptor
    TT[i][:,17]=0

# create chemistry family name as feature
CFN=np.zeros((111,3)) 
for i in range(0,111):   

    if FILE[i][-2:]=='MV':
        CFN[i,:]=np.array([1,0,0])
    if FILE[i][-2:]=='CV':
        CFN[i,:]=np.array([0,1,0])
    if FILE[i][-2:]=='AV':
        CFN[i,:]=np.array([0,0,1])

# read already extracted treelet histogram by another code

mat = loadmat('treeletcobra.mat')
GX=mat['GX']



# get the atomic features
AF=np.zeros((111,200)) 
RAF=np.zeros((111,50)) 
MAF=np.zeros((111,50))  
for i in range(0,111):   
    AF[i,:]=np.hstack((np.min(TT[i],0),np.max(TT[i],0),np.mean(TT[i],0),np.std(TT[i],0)))    
    MAF[i,:]=  np.max(TT[i],0)
    RAF[i,:]=  TT[i][int(react[i])-1,:]

# create inputs of model according to selection
if feature==0:
    if fmcode==1:
        X=np.hstack((F,CFN))
    else:
        X=F
elif feature==1:
    X=AF
elif feature==2:
    X=GX
elif feature==3:
    if fmcode==1:
        X=np.hstack((F,CFN,AF))
    else:
        X=np.hstack((F,AF))

elif feature==4:
    if fmcode==1:
        X=np.hstack((F,CFN,AF,GX))
    else:
        X=np.hstack((F,AF,GX))

elif feature==5:
    if fmcode==1:
        X=np.hstack((F,CFN,RAF,GX))
    else:
        X=np.hstack((F,RAF,GX))
elif feature==6:
    if fmcode==1:
        X=np.hstack((F,CFN,MAF,GX))
    else:
        X=np.hstack((F,MAF,GX))
elif feature==7:
    X=RAF
else:
    if fmcode==1:
        X=np.hstack((F,CFN,RAF))
    else:
        X=np.hstack((F,RAF))

 

X=X[:,np.where(X.std(0)>0)[0]]

Xorg=X.copy()
Yorg=Y.copy()

#X=(X-X.mean(0))/X.std(0)
#Y=(Y-Y.mean(0))/Y.std(0)

# do outsample test by 10-fold cross validation for 100 times
C=[];M=[];D=[];SP=[];cSP=[]

for iter in range(0,nsim): # loop over repetition of test
    
    p=np.random.permutation(111)
    y_pred=np.zeros((111))
    shap_values=np.zeros((0,X.shape[1]))
    xx=np.zeros((0,X.shape[1]))
    
    for i in range(0,110,11): # loop over k-fold cross validation
        
        # make which element is in test which is train
        I=np.zeros(111)
        I[p[i:i+11]]=1
        if i==99:
            I[p[i+11]]=1
        # creat gradient boosting decision tree 
        #regr = xgb.XGBRegressor( nthread=4,colsample_bytree=0.2, gamma=0.0, learning_rate=0.01, max_depth=4, min_child_weight=1.5, n_estimators=10000, reg_alpha=0.9, reg_lambda=0.6, subsample=0.2, seed=seed, silent=1)

        # train the model by train set
        #regr.fit(X[np.where(I==0)[0],:], Y[np.where(I==0)[0]])
        #kernel = gp.kernels.ConstantKernel(1.0, (1e-5, 1e5)) * gp.kernels.RBF(1.0, (1e-5, 1e5))
        kernel = gp.kernels.DotProduct() #+ gp.kernels.WhiteKernel()
        regr = gp.GaussianProcessRegressor(kernel=kernel,alpha=10.0) #n_restarts_optimizer=10, alpha=10.0, normalize_y=False)

        
        xstd =Xorg[np.where(I==0)[0],:].std(0)
        X=Xorg[:,np.where(xstd>0.00)[0]]
        xstd =X[np.where(I==0)[0],:].std(0)
        xmean=X[np.where(I==0)[0],:].mean(0)

        X=(X-xmean)/xstd

        ymean=Yorg[np.where(I==0)[0]].mean(0)
        ystd=Yorg[np.where(I==0)[0]].std(0)
        Y=(Yorg-ymean)/ystd


        regr.fit(X[np.where(I==0)[0],:], Y[np.where(I==0)[0]])

        # take the test result for test set
        yhat,pconv = regr.predict(X[np.where(I==1)[0],:],return_std=True)

        y_pred[np.where(I==1)[0]]=(ystd*yhat+ymean)

        if isshap==1:
            xtran=X
            explainer = shap.TreeExplainer(regr)
            shap_values = np.vstack( (shap_values,explainer.shap_values(xtran)))
            xx= np.vstack( (xx,xtran))

    if isshap==1:
        cSP.append(np.abs(shap_values).mean(axis=0))
        SP.append(shap_values)
        shap.summary_plot(shap_values, xx)
        df = pd.DataFrame(data=np.array(cSP).T)
        df.to_csv('outputs/gbdt_shapvalues_'+str(feature)+'_.csv')
    r2=r2_score(Yorg,y_pred)
    if r2<0.9:
        print(r2)
        continue  # pass #
    C.append(r2_score(Yorg,y_pred))
    M.append(np.mean(np.abs(Yorg-y_pred)))    
    D.append(y_pred)
        
    print('so far R2=', str(np.mean(C)),'±', str(np.std(C)), ' MAE=', str(np.mean(M)) ,'±', str(np.std(M)))

df = pd.DataFrame(data=np.array(D).T)
df.to_csv('outputs/gpr_prediction_'+str(feature)+'_'+str(fmcode)+'_.csv')



    
