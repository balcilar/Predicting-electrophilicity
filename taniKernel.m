function K=taniKernel(X,Y)
for i=1:size(X,1)
    for j=1:size(Y,1)
        K(i,j)=(X(i,:)*Y(j,:)')/(X(i,:)*X(i,:)'+Y(j,:)*Y(j,:)'-X(i,:)*Y(j,:)');
        %K(i,j)=sum(min([X(i,:);Y(j,:)]))/sum(max([X(i,:);Y(j,:)]));
    end
end