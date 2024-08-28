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
YY=X(:,sel_f+1);%----�������ǩ
Y=[X(:,1),YY];%----�������ǩ
[kk,k]=size(Y);
indices=crossvalind('Kfold',Y(1:kk,1),5);
%%%------------------indices��һ��Nά��������ÿ��Ԫ�ض�Ӧ��ֵΪ�õ�Ԫ�����İ��ı��
Ksum = 0;
% L_A=zeros(nn,nn);%���ڴ�ż������ŷʽ����

for i= 1:5
test = (indices ==i); %���test��Ԫ�������ݼ��ж�Ӧ�ĵ�Ԫ���
train = ~test;%train��Ԫ�صı��Ϊ��testԪ�صı��
train_data=Y(train,:);%�����ݼ��л��ֳ�train����������

test_data=Y(test,:);%test������


%%%%%%-------------------------------ȫ����������ѵ��ģ��

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
for ii=1:nx %��������
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

%----������ྫ��
if isempty(test_label1)==0
    acc=length(find(predicted_labels==test_label1))/nx;
end
    Ksum = Ksum +acc;
end
%ѵ����5����֤ʱ��ȡƽ������ֵ
a = Ksum/5;%-------׼ȷ��
end








