function [R2 MAE]=linearreg_analysis(GF,Y,kfold,nsim)
n=size(GF,1);

for i=1:nsim
    
    p=randperm(n);
    l=round(linspace(1,n+1,kfold+1));
    yhat=zeros(n,1);
    for k=1:kfold
        I=zeros(n,1);
        I(p(l(k):l(k+1)-1))=1;
        
        x=GF(I==0,:);
        y=Y(I==0,:);
        x(:,end+1)=1;
        a=pinv(x)*y;
        
        x=GF(I==1,:);
        x(:,end+1)=1;
        yhat(I==1,1)=x*a;
    end
    
    R2(i)=corr(yhat,Y).^2;
    MAE(i)=mean(abs(yhat-Y));
end