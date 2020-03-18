function YPCA=pca_prediction(Xtrain,Ytrain,Xtest,pcavar)
p=randperm(size(Xtrain,1));
ntr=round(size(Xtrain,1)*0.8);
for i=1:length(pcavar)
    xtrain=Xtrain(p(1:ntr),:);
    xval=Xtrain(p(ntr+1:end),:);
    ytrain=Ytrain(p(1:ntr));
    yval=Ytrain(p(ntr+1:end));
    [coeff,score,latent,tsquared,explained,mu] = pca(xtrain);

    cs=cumsum(explained);
    ind=find(cs<pcavar(i)*100);
    if isempty(ind)
        ind=1;
    end
    trX=(xtrain)*coeff(:,1:ind(end));
    tsX=(xval)*coeff(:,1:ind(end));

    trX=[trX ones(size(trX,1),1)];
    tsX=[tsX ones(size(tsX,1),1)];

    X=pinv(trX)*ytrain;
    YPCA(i)=mean(abs(tsX*X-yval));
end
[u v]=min(YPCA);

pcavar=pcavar(v);    






[coeff,score,latent,tsquared,explained,mu] = pca(Xtrain);

cs=cumsum(explained);
ind=find(cs<pcavar*100);
if isempty(ind)
    ind=1;
end
trX=(Xtrain)*coeff(:,1:ind(end));
tsX=(Xtest)*coeff(:,1:ind(end));

trX=[trX ones(size(trX,1),1)];
tsX=[tsX ones(size(tsX,1),1)];

X=pinv(trX)*Ytrain;
YPCA=tsX*X;
