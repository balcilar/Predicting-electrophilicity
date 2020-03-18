function [R2 MAE]=pcareg_analysis(GF,Y,kfold,nsim,pcavar)

%zA=zscore(GF);

n=size(GF,1);

for i=1:nsim    
    p=randperm(n);
    l=round(linspace(1,n+1,kfold+1));
    yhat=zeros(n,1);
    for k=1:kfold
        I=zeros(n,1);
        I(p(l(k):l(k+1)-1))=1; 
        
        trX=GF(I==0,:);
        
        sd=std(trX);
        GFF=GF(:,sd>0.01);
        trX=GFF(I==0,:);
        
        zA=(GF(:,sd>0.01)-mean(trX))./std(trX);
            
            
        
        x=zA(I==0,:);
        y=Y(I==0,:);        
        yhat(I==1,1)=pca_prediction(zA(I==0,:),Y(I==0,:),zA(I==1,:),pcavar);       
        
    end
    
    R2(i)=corr(yhat,Y).^2;
    MAE(i)=mean(abs(yhat-Y));
end
I=find(R2<0.5);
R2(I)=[];MAE(I)=[];