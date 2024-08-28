function acc=Test_Acc(x)
%we used this function to calculate the accuracy 
global train_data test_data
x=x>0.6;
x=cat(2,x,zeros(size(x,1),1));
x=logical(x);

if sum(x)==0
    y=inf;
    return;
end
[acc,testY_est] = knnPrediction( train_data(:,x), train_data(:,end), test_data(:,x) , test_data(:,end),5);
acc = 100*(1-acc);


