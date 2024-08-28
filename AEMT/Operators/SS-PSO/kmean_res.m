function [a,jie_sum]=kmean_res(pp,X,u,k,mm)

sel_f=[];
for ii=1:mm
    if pp(ii)~=0
        sel_f=[sel_f,u{ii}(pp(ii))];
    end
end
% 
%------------------------------------------SVM
jie_sum=size(sel_f,2);
if jie_sum==0
    a=0;
else


% nx=size(X,1);
YY=X(:,sel_f+1);%----不含类标签
Y=[X(:,1),YY];%----加上类标签
[kk,k]=size(Y);
indices=crossvalind('Kfold',Y(1:kk,1),5);
%%%------------------indices是一个N维列向量，每个元素对应的值为该单元所属的包的编号
Ksum = 0;
% L_A=zeros(nn,nn);%用于存放计算过的欧式距离

for i= 1:5
test = (indices ==i); %获得test集元素在数据集中对应的单元编号
train = ~test;%train集元素的编号为非test元素的编号
train_data=Y(train,:);%从数据集中划分出train样本的数据

test_data=Y(test,:);%test样本集


%%%%%%-------------------------------全部特征进行训练模型

X_train=train_data;
Y_test=test_data;

train_data1=X_train(:,2:k);
train_label1=X_train(:,1);
test_data1=Y_test(:,2:k);
test_label1=Y_test(:,1);
%-------------------3NN
K=3;
nx=size(test_data1,1);
ny=size(train_data1,1);
 Dis=zeros(nx,ny);
 ind=zeros(nx,ny);
for ii=1:nx %测试样本
    P=test_data1(ii,:);
    Q=train_data1;
    

    Dis(ii,:)=sqrt(P.^2*ones(size(Q'))+ones(size(P))*(Q').^2-2*P*Q');
    [Dis(ii,:),ind(ii,:)]=sort(Dis(ii,:));
end
k_nn=ind(:,1:K);
nn_index=k_nn(:,1);

predicted_labels=zeros(nx,1);
for l_i=1:nx
    options=unique(train_label1(k_nn(l_i,:)'));
    max_count=0;
    max_label=0;
    for j=1:length(options)
        L=length(find(train_label1(k_nn(l_i,:)')==options(j)));
        if L>max_count
            max_label=options(j);
            max_count=L;
        end
    end
    predicted_labels(l_i)=max_label;
end

%----计算分类精度
if isempty(test_label1)==0
    acc=length(find(predicted_labels==test_label1))/nx;
end
    Ksum = Ksum +acc;
end
%训练集5折验证时，取平均精度值
a = Ksum/5;%-------准确率
end








