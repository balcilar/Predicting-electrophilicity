function [R2 MAE]=svr_rbfkernel_analysis(GF,Y,kfold,nsim,C,P,S,norm)
n=size(GF,1);

for i=1:nsim    
    p=randperm(n);
    l=round(linspace(1,n+1,kfold+1));
    yhat=zeros(n,1);
    for k=1:kfold
        I=zeros(n,1);
        I(p(l(k):l(k+1)-1))=1; 
        if norm==0 % no normalization
            GFF=GF;
        elseif norm==-1 %full normalization
            trX=GF(I==0,:);
            sd=std(trX);
            GFF=GF(:,sd>0.01);
            trX=GFF(I==0,:);
            GFF=(GFF-mean(trX))./std(trX);        
        else
            gf=GF(:,1:end-norm);
            trX=gf(I==0,:);
            sd=std(trX);
            GFF=gf(:,sd>0.01);
            trX=GFF(I==0,:);
            GFF=(GFF-mean(trX))./std(trX); 
            
            gf=GF(:,end-norm+1:end);
            trX=gf(I==0,:);
            sd=max(trX);
            gf=gf(:,sd>0.01);
            trX=gf(I==0,:);
            gf=gf./max(max(trX));
            
            GFF=[GFF gf];            
        end
        
        yhat(I==1,:)=svr_rbfkernel_prediction(GFF(I==0,:),Y(I==0,:),GFF(I==1,:),C,P,S);
    end
    
    R2(i)=corr(yhat,Y).^2;
    MAE(i)=mean(abs(yhat-Y));
end
I=find(R2<0.60);
R2(I)=[];MAE(I)=[];

