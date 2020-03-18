clear
close all
warning off
load cobradat

% compile libsvm
cd libsvm-3.24/matlab
make
cd ../..
addpath('libsvm-3.24/matlab');


clc

% if chemistry family code is included or not.
fmcode=1;

% first 16 columns are global features
GF=F(:,1:16);
% last column is output value
Y=F(:,end);



if fmcode==1

    % 17-18-19 columns are family code
    GF(:,17:19)=0;

    n=length(FILE);
    mv=0;cv=0;av=0;
    for i=1:n
        if strcmp(FILE{i}(end-1:end),'MV')
            mv=mv+1;
            GF(i,17)=1;
        elseif strcmp(FILE{i}(end-1:end),'AV')
            av=av+1;
            GF(i,18)=1;
        elseif strcmp(FILE{i}(end-1:end),'CV')
            cv=cv+1;
            GF(i,19)=1;
        end
    end
end

% make atomic feature table
AF=zeros(111,200);
RAF=zeros(111,50);
MAF=zeros(111,50);

for i=1:111
    af=TT{i}(:,5:end);
    % remove problematic atomic feature
    af(:,18)=0;
    AF(i,:)=[mean(af) max(af) min(af) std(af)];  
    reac=F(i,end-1);
    RAF(i,:)=af(reac,:);
    MAF(i,:)=max(af);
end

% read treelet histograms into GX
load treeletcobra
GX=double(GX);

CR=[];
FF=[GF AF RAF GX];
for i=1:size(FF,2)
    CR(i,1)=corr(Y,FF(:,i));
    CR(i,2)=i;
end
CR(isnan(CR(:,1)),:)=[];
CR=sortrows(abs(CR),-1);

for i=1:16
    names{i}=['global ' Vnames{i}];
end
names{17}='MV';names{18}='AV';names{19}='CV';

for i=1:50
    names{i+19}=['mean ' Anames{i+4}];
    names{i+19+50}=['max ' Anames{i+4}];
    names{i+19+100}=['min ' Anames{i+4}];
    names{i+19+150}=['std ' Anames{i+4}];
    names{i+19+200}=['reac ' Anames{i+4}];
end

xlbl{1}='';
for i =1:50  
    xlbl{i+1}=names{ CR(i,2)};
end

figure;bar(CR(1:50,1),0.4)
xticklabels(xlbl)
xtickangle(90)
ylim([0.6 1])
    
    
  

% 10 fold cross validation for global inputs under rbf kernel svr
C=[100 1000];P=[0.2 1];S=[1e-4 1e-2 1];
[R2 MAE]=svr_rbfkernel_analysis(GF,Y,10,50,C,P,S,-1);
fprintf('10 fold cross validation for global inputs under rbf kernel svr \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]


% 10 fold cross validation for atomic inputs under rbf kernel svr
C=[0.1 1 10 100 1000 10000];
P=[0.1 0.2 0.5 1 5 10];
S=[1e-5 1e-4 1e-3 1e-2];
[R2 MAE]=svr_rbfkernel_analysis(AF,Y,10,50,C,P,S,-1);
fprintf('10 fold cross validation for atomic inputs under rbf kernel svr \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]

% 10 fold cross validation for reactive atomic inputs under rbf kernel svr
C=[0.1 1 10 100 1000 10000];
P=[0.1 0.2 0.5 1 5 10];
S=[1e-5 1e-4 1e-3 1e-2];
[R2 MAE]=svr_rbfkernel_analysis(RAF,Y,10,50,C,P,S,-1);
fprintf('10 fold cross validation for reactive atomic inputs under rbf kernel svr \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]

% 10 fold cross validation for reactive atomic+global inputs under rbf kernel svr
C=[0.1 1 10 100 1000 10000];
P=[0.1 0.2 0.5 1 5 10];
S=[1e-5 1e-4 1e-3 1e-2];
[R2 MAE]=svr_rbfkernel_analysis([GF RAF],Y,10,50,C,P,S,-1);
fprintf('10 fold cross validation for reactive atomic+global inputs under rbf kernel svr \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]



% 10 fold cross validation for topologic inputs under rbf kernel svr
C=[100 1000];P=[0.2 1];S=[1e-4 1e-2 1];
[R2 MAE]=svr_rbfkernel_analysis(GX,Y,10,50,C,P,S,0);
fprintf('10 fold cross validation for topologic inputs under rbf kernel svr \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]



% 10 fold cross validation for global+atomic inputs under rbf kernel svr
C=[0.1 1 10 100 1000 10000];
P=[0.1 0.2 0.5 1 5 10];
S=[1e-5 1e-4 1e-3 1e-2];
[R2 MAE]=svr_rbfkernel_analysis([GF AF],Y,10,50,C,P,S,-1);
fprintf('10 fold cross validation for global+atomic inputs under rbf kernel svr \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]


% 10 fold cross validation for global+atomic+topologic inputs under rbf kernel svr
C=[0.1 1 10 100 1000 10000];
P=[0.1 0.2 0.5 1 5 10];
S=[1e-5 1e-4 1e-3 1e-2];
[R2 MAE]=svr_rbfkernel_analysis([GF AF GX],Y,10,50,C,P,S,-1);
fprintf('10 fold cross validation for global+atomic+topologic inputs under rbf kernel svr \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]


% 10 fold cross validation for global+reactive atomic+topologic inputs under rbf kernel svr
C=[0.1 1 10 100 1000 10000];
P=[0.1 0.2 0.5 1 5 10];
S=[1e-5 1e-4 1e-3 1e-2];
[R2 MAE]=svr_rbfkernel_analysis([GF RAF GX],Y,10,50,C,P,S,-1);
fprintf('10 fold cross validation for global+reactiveatomic+topologic inputs under rbf kernel svr \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]


% 10 fold cross validation for global+max atomic+topologic inputs under rbf kernel svr
C=[0.1 1 10 100 1000 10000];
P=[0.1 0.2 0.5 1 5 10];
S=[1e-5 1e-4 1e-3 1e-2];
[R2 MAE]=svr_rbfkernel_analysis([GF MAF GX],Y,10,50,C,P,S,-1);
fprintf('10 fold cross validation for global+max atomic+topologic inputs under rbf kernel svr \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]





% 10 fold cross validation for global inputs under tani kernel svr
C=[0.1 1 10 100 1000 10000 100000 500000];
P=[0.0001 0.001 0.1 0.2 0.5 1 5 10];
[R2 MAE]=svr_tanikernel_analysis(GF,Y,10,50,C,P,-1);
fprintf('10 fold cross validation for global inputs under tani kernel svr \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]

% 10 fold cross validation for atomic inputs under tani kernel svr
C=[0.1 1 10 100 1000 10000 100000 500000];
P=[0.0001 0.001 0.1 0.2 0.5 1 5 10];
[R2 MAE]=svr_tanikernel_analysis(AF,Y,10,50,C,P,-1);
fprintf('10 fold cross validation for atomic inputs under tani kernel svr \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]

% 10 fold cross validation for reactive atomic inputs under tani kernel svr
C=[0.1 1 10 100 1000 10000 100000 500000];
P=[0.0001 0.001 0.1 0.2 0.5 1 5 10];
[R2 MAE]=svr_tanikernel_analysis(RAF,Y,10,50,C,P,-1);
fprintf('10 fold cross validation for reactive atomic inputs under tani kernel svr \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]


% 10 fold cross validation for reactive atomic+global inputs under tani kernel svr
C=[0.1 1 10 100 1000 10000 100000 500000];
P=[0.0001 0.001 0.1 0.2 0.5 1 5 10];
[R2 MAE]=svr_tanikernel_analysis([GF RAF],Y,10,50,C,P,-1);
fprintf('10 fold cross validation for reactive atomic+global inputs under tani kernel svr \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]



% 10 fold cross validation for topologic inputs under tani kernel svr
C=[0.1 1 10 100 1000 10000 100000 500000];
P=[0.0001 0.001 0.1 0.2 0.5 1 5 10];
[R2 MAE]=svr_tanikernel_analysis(GX,Y,10,50,C,P,0);
fprintf('10 fold cross validation for topologic inputs under tani kernel svr \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]



% 10 fold cross validation for global+atomic inputs under tani kernel svr
C=[0.1 1 10 100 1000 10000 100000 500000];
P=[0.0001 0.001 0.1 0.2 0.5 1 5 10];
[R2 MAE]=svr_tanikernel_analysis([AF GF],Y,10,50,C,P,-1);
fprintf('10 fold cross validation for global+atomic inputs under tani kernel svr \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]

% 10 fold cross validation for global+atomic+topologic inputs under tani kernel svr
C=[0.1 1 10 100 1000 10000 100000 500000];
P=[0.0001 0.001 0.1 0.2 0.5 1 5 10];
[R2 MAE]=svr_tanikernel_analysis([AF GF GX],Y,10,50,C,P,size(GX,2));
fprintf('10 fold cross validation for global+atomic+topologic inputs under tani kernel svr \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]

% 10 fold cross validation for global+reactive atomic+topologic inputs under tani kernel svr
C=[0.1 1 10 100 1000 10000 100000 500000];
P=[0.0001 0.001 0.1 0.2 0.5 1 5 10];
[R2 MAE]=svr_tanikernel_analysis([GF RAF GX],Y,10,50,C,P,size(GX,2));
fprintf('10 fold cross validation for global+reactive atomic+topologic inputs under tani kernel svr \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]


% 10 fold cross validation for global+max atomic+topologic inputs under tani kernel svr
C=[0.1 1 10 100 1000 10000 100000 500000];
P=[0.0001 0.001 0.1 0.2 0.5 1 5 10];
[R2 MAE]=svr_tanikernel_analysis([GF MAF GX],Y,10,50,C,P,size(GX,2));
fprintf('10 fold cross validation for global+max atomic+topologic inputs under tani kernel svr \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]





% 10 fold cross validation for global inputs under linear regression
[R2 MAE]=linearreg_analysis(GF,Y,10,100);
fprintf('10 fold cross validation for global inputs under linear regression\n');
[mean(MAE) std(MAE) mean(R2) std(R2)]


% 10 fold cross validation for reactive atomic inputs under linear regression
[R2 MAE]=linearreg_analysis(RAF,Y,10,100);
fprintf('10 fold cross validation for reactive atomic inputs under linear regression\n');
[mean(MAE) std(MAE) mean(R2) std(R2)]


% 10 fold cross validation for globql inputs under pca regression
pcavar=[0.6 0.7 0.8 0.9 0.95 0.96 0.97 0.98 0.99 0.999];
[R2 MAE]=pcareg_analysis(GF,Y,10,50,pcavar);
fprintf('10 fold cross validation for globql inputs under pca regression \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]


% 10 fold cross validation for atomic inputs under pca regression
pcavar=[0.6 0.7 0.8 0.9 0.95 0.96 0.97 0.98 0.99 0.999];
[R2 MAE]=pcareg_analysis(AF,Y,10,50,pcavar);
fprintf('10 fold cross validation for atomic inputs under pca regression \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]


% 10 fold cross validation for reactive atomic inputs under pca regression
pcavar=[0.6 0.7 0.8 0.9 0.95 0.96 0.97 0.98 0.99 0.999];
[R2 MAE]=pcareg_analysis(RAF,Y,10,50,pcavar);
fprintf('10 fold cross validation for reactive atomic inputs under pca regression \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]


% 10 fold cross validation for reactive atomic+global inputs under pca regression
pcavar=[0.6 0.7 0.8 0.9 0.95 0.96 0.97 0.98 0.99 0.999];
[R2 MAE]=pcareg_analysis([GF RAF],Y,10,50,pcavar);
fprintf('10 fold cross validation for reactive atomic+global inputs under pca regression \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]



% 10 fold cross validation for topologic inputs under pca regression
pcavar=[0.6 0.7 0.8 0.9 0.95 0.96 0.97 0.98 0.99 0.999];
[R2 MAE]=pcareg_analysis(GX,Y,10,50,pcavar);
fprintf('10 fold cross validation for topologic inputs under pca regression \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]


% 10 fold cross validation for global+atomic inputs under pca regression
pcavar=[0.6 0.7 0.8 0.9 0.95 0.96 0.97 0.98 0.99 0.999];
[R2 MAE]=pcareg_analysis([GF AF],Y,10,50,pcavar);
fprintf('10 fold cross validation for global+atomic inputs under pca regression \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]


% 10 fold cross validation for global+atomic+topologic inputs under pca regression
pcavar=[0.6 0.7 0.8 0.9 0.95 0.96 0.97 0.98 0.99 0.999];
[R2 MAE]=pcareg_analysis([GF AF GX],Y,10,50,pcavar);
fprintf('10 fold cross validation for global+atomic+topologic inputs under pca regression \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]


% 10 fold cross validation for global+reactiveatomic+topologic inputs under pca regression
pcavar=[0.6 0.7 0.8 0.9 0.95 0.96 0.97 0.98 0.99 0.999];
[R2 MAE]=pcareg_analysis([GF RAF GX],Y,10,50,pcavar);
fprintf('10 fold cross validation for global+reactive atomic+topologic inputs under pca regression \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]

% 10 fold cross validation for global+max atomic+topologic inputs under pca regression
pcavar=[0.6 0.7 0.8 0.9 0.95 0.96 0.97 0.98 0.99 0.999];
[R2 MAE]=pcareg_analysis([GF MAF GX],Y,10,50,pcavar);
fprintf('10 fold cross validation for global+max atomic+topologic inputs under pca regression \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]



% 10 fold cross validation for global inputs under lasso regression
Lambda=0.15:0.05:0.5;
[R2 MAE]=lassreg_analysis(GF,Y,10,50,Lambda);
fprintf('10 fold cross validation for global inputs under lasso regression \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]


% 10 fold cross validation for atomic inputs under lasso regression
Lambda=0.15:0.05:0.5;
[R2 MAE]=lassreg_analysis(AF,Y,10,50,Lambda);
fprintf('10 fold cross validation for atomic inputs under lasso regression \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]


% 10 fold cross validation for reactive atomic inputs under lasso regression
Lambda=0.10:0.05:0.5;
[R2 MAE]=lassreg_analysis(RAF,Y,10,50,Lambda);
fprintf('10 fold cross validation for reactive atomic inputs under lasso regression \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]


% 10 fold cross validation for reactive atomic+global inputs under lasso regression
Lambda=0.10:0.05:0.5;
[R2 MAE]=lassreg_analysis([GF RAF],Y,10,50,Lambda);
fprintf('10 fold cross validation for reactive atomic+global inputs under lasso regression \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]



% 10 fold cross validation for topologic inputs under lasso regression
Lambda=0.15:0.05:0.5;
[R2 MAE]=lassreg_analysis(GX,Y,10,50,Lambda);
fprintf('10 fold cross validation for topologic inputs under lasso regression \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]


% 10 fold cross validation for global+atomic inputs under lasso regression
Lambda=0.15:0.05:0.5;
[R2 MAE]=lassreg_analysis([GF AF],Y,10,50,Lambda);
fprintf('10 fold cross validation for global+atomic inputs under lasso regression \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]


% 10 fold cross validation for global+atomic+topologic inputs under lasso regression
Lambda=0.15:0.05:0.5;
[R2 MAE]=lassreg_analysis([GF AF GX],Y,10,50,Lambda);
fprintf('10 fold cross validation for global+atomic+topologic inputs under lasso regression \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]


% 10 fold cross validation for global+reactive atomic+topologic inputs under lasso regression
Lambda=0.15:0.05:0.5;
[R2 MAE]=lassreg_analysis([GF RAF GX],Y,10,50,Lambda);
fprintf('10 fold cross validation for global+reactive atomic+topologic inputs under lasso regression \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]

% 10 fold cross validation for global+max atomic+topologic inputs under lasso regression
Lambda=0.15:0.05:0.5;
[R2 MAE L B]=lassreg_analysis([GF MAF GX],Y,10,50,Lambda);
fprintf('10 fold cross validation for global+max atomic+topologic inputs under lasso regression \n');
[mean(MAE) std(MAE) mean(R2) std(R2)]

% error histogram

pred=csvread('outputs/gbdt_10fold_maxatm_glob_topology_predictions.csv');
E=F(:,end)-pred;

hist(E(:),100)




% linear regression according to top features according to gdbt model
sg=csvread('outputs/gbdt_10fold_maxatm_glob_topology_significance.csv');
sg(1,:)=[];sg(:,1)=[];
sgs=mean(abs(sg'));
sgs=sortrows([sgs' [1:933]'],-1);

X=[GF MAF GX];
MAE=zeros(50,100);
R2=zeros(50,100);
for i=1:50
    XX=X(:,sgs(1:i,2));
    % 10 fold cross validation for top 20 features under linear regression
    [R2(i,:) MAE(i,:)]=linearreg_analysis(XX,Y,10,100);
    i
end
fanChart(1:50,MAE)
hold on;plot(mean(MAE'),'b-')



