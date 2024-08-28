function [train_data test_data] = divide_data(X,Y)
rng('default')
rng(0)
dataset = [X Y];
train_data = [];
test_data = [];
class = unique(Y);
num_of_class = length(class);
for i = 1:num_of_class
    select(Y~=class(i,:))=0;
    select(Y==class(i,:))=1;
    select = logical(select);
    instance = dataset(select,:);
    r = randperm(size(instance,1));
    trn = r(1:floor(0.7*length(r)));
    vald = r(floor(0.7*length(r))+1:end);
    progress_trn = instance(trn,:);
    progress_vald = instance(vald,:);
    train_data = [train_data;progress_trn];
    test_data = [test_data;progress_vald];      
end
end