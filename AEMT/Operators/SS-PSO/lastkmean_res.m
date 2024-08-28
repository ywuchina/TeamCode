function [ary,jie_sum]=lastkmean_res(pp,X,Y,u,mm,f)

sel_f=[];
for ii=1:mm
    if pp(ii)~=0
        sel_f=[sel_f,u{ii}(pp(ii))];
    end
end
% 
jie_sum=size(sel_f,2);

sel_fea=[];
for i=1:jie_sum
    sel_fea=[sel_fea,f(sel_f(i))];
end

%------------------------------------------SVM
X1=X(:,sel_fea+1);%----不含类标签，训练集
Y1=Y(:,sel_fea+1);%----不含类标签，测试集
X2=[X(:,1),X1];%----加上类标签
Y2=[Y(:,1),Y1];%----加上类标签
[kk,k]=size(Y2);
train_data1=X2(:,2:k);
train_label1=X2(:,1);
test_data1=Y2(:,2:k);
test_label1=Y2(:,1);

mdl = fitcknn(train_data1,train_label1,'NumNeighbors',3,'Standardize',1);
       c = predict(mdl,test_data1); 

    correct=0;
    n1=size(test_label1,1);
for ii=1:n1
    if test_label1(ii)==c(ii)
        correct=correct+1;
    end
end
    ary=correct/n1;






