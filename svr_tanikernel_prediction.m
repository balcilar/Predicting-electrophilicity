function [ypred K KK param,yhat2]=svr_tanikernel_prediction(Xtrain,Ytrain,Xtest,C,P)


    p=randperm(size(Xtrain,1));
    ntr=round(size(Xtrain,1)*0.8);

    trainData=Xtrain(p(1:ntr),:);
    valData=Xtrain(p(ntr+1:end),:);
    ytrain=Ytrain(p(1:ntr));
    yval=Ytrain(p(ntr+1:end));
    numTrain=size(trainData,1);
    numVal=size(valData,1);
    
                    
    K =  [ (1:numTrain)' , taniKernel(trainData,trainData) ];
    KK = [ (1:numVal)'  , taniKernel(valData,trainData)  ];
                
                

    itr=0;
    for i =1:length(C)
        for j =1:length(P)
            
               itr=itr+1;

                                
                
                param=['-s 3 -t 4 -c ' num2str(C(i)) ' -p ' num2str(P(j)) ' -q'];

                model = svmtrain(ytrain, K, param);
                [ypred, acc, decVals] = svmpredict(yval, KK, model,'-q');
                yhat2(itr,:)=[mean(abs(ypred-yval)) i j];
           
        end
    end
    yhat2=sortrows(yhat2,1);
    
    K =  [ (1:size(Xtrain,1))' , taniKernel(Xtrain,Xtrain) ];
    KK = [ (1:size(Xtest,1))'  , taniKernel(Xtest,Xtrain)  ];   
    
    param=['-s 3 -t 4 -c ' num2str(C(yhat2(1,2))) ' -p ' num2str(P(yhat2(1,3))) ' -q'];

    model = svmtrain(Ytrain, K, param);
    [ypred, acc, decVals] = svmpredict(zeros(size(Xtest,1),1), KK, model,'-q');


            
            


   