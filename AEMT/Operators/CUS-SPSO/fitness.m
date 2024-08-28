function y = fitness(x)
%We used this function to calculate the fitness value of Task 1.
global train_data;
[m,n] = size(train_data);
c = unique(train_data(:,end));
uc = length(c);
Sz = 0.1;
Ksum = 0;
k = 5;
x = x > 0.6;
output_pre = [];
output_act = [];
x = cat(2,x,zeros(size(x,1),1));
x = logical(x);
if sum(x) == 0
    y = inf;
    return;
end
indices = crossvalind('Kfold',m,k);
for i= 1:k
    test = (indices == i);
    train = ~test;
    [acc, testYest] = knnPrediction(train_data(train,x),train_data(train,end),train_data(test,x),train_data(test,end),5);
    Ksum = Ksum + acc; 
    output_pre = [output_pre,testYest];
    output_act = [output_act;train_data(test,end)];
end

for i = 1:uc
    uq(output_act~=c(i,:))=0;
    uq(output_act==c(i,:))=1;
    uq = logical(uq);
    ek = output_pre(:,uq);
    ec = find(ek==c(i,:));
    length_ec = length(ec);
    length_uq = sum(uq);
    ratio(1,i) = 1-length_ec/length_uq;
end
acc = (1/uc)*sum(ratio);
y = (1-Sz)*acc+Sz*sum(x)/(length(x)-1);







