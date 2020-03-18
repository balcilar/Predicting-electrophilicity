function [YLAS lam B1]=lasso_prediction(Xtrain,Ytrain,Xtest,lam)

p=randperm(size(Xtrain,1));
ntr=round(size(Xtrain,1)*0.8);
for i=1:length(lam)
    xtrain=Xtrain(p(1:ntr),:);
    xval=Xtrain(p(ntr+1:end),:);
    ytrain=Ytrain(p(1:ntr));
    yval=Ytrain(p(ntr+1:end));
    
    [B1 FitInfo1] = lasso(xtrain,ytrain,'Lambda',lam(i));
    that=xtrain*B1;
    that(:,2)=1;
    a=pinv(that)*ytrain;
    tmp=xval*B1;
    tmp(:,2)=1;
    
    yhat(i)=mean(abs(tmp*a-yval));
end
[u v]=min(yhat);

lam=lam(v);

[B1 FitInfo1] = lasso(Xtrain,Ytrain,'Lambda',lam);
that=Xtrain*B1;
that(:,2)=1;
a=pinv(that)*Ytrain;
tmp=Xtest*B1;
tmp(:,2)=1;
YLAS=tmp*a; 
        
    