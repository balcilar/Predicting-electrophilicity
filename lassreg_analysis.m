function [R2 MAE L B]=lassreg_analysis(GF,Y,kfold,nsim,lam)

L=[];
B=[];
n=size(GF,1);

for i=1:nsim    
    p=randperm(n);
    l=round(linspace(1,n+1,kfold+1));
    yhat=zeros(n,1);
    for k=1:kfold
        I=zeros(n,1);
        I(p(l(k):l(k+1)-1))=1; 
        [YLAS Lam B1] =lasso_prediction(GF(I==0,:),Y(I==0,:),GF(I==1,:),lam);
        yhat(I==1,:)=YLAS;
        L=[L;Lam];
        B=[B B1];
        
    end
    
    R2(i)=corr(yhat,Y).^2;
    MAE(i)=mean(abs(yhat-Y));
end
I=find(R2<0.6);
R2(I)=[];MAE(I)=[];