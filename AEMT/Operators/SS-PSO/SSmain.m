
function [Accuracy,d_feature,tt1,G,Time_train1,AA_AC]=SSmain(PF,label,prm) 


t0=clock;
fly=1;
XX=data_t(fly);

[kk,k]=size(XX);
NUM=2;
indices=crossvalind('Kfold',XX(1:kk,1),NUM);
%%%%--------------------分别对每一份测试集进行验证，最后十次平均数为最终的测试集的分类精度
G=cell(NUM,1);
a=cell(NUM,1);
d=cell(NUM,1);
AA_AC=cell(NUM,1);
Time_train=[];
for i=1:NUM
 test = (indices ==i); %获得test集元素在数据集中对应的单元编号
 train = ~test;%train集元素的编号为非test元素的编号
 train_data=XX(train,:);%从数据集中划分出train样本的数据
 test_data=XX(test,:);%test样本集
 X_train=train_data;

 Y_test=test_data;

 [gbest,u,mm,f,tt_train,AC]=main2(X_train);
 ggbest=evaluationlast(gbest,mm,u,X_train,Y_test,f);

G{i}=ggbest;
a{i}=ggbest(1,mm+1);
d{i}=ggbest(1,mm+2);
 Time_train=[Time_train;tt_train];
 AA_AC{i}=AC;
end
a1=0;d1=0;Time_train1=0;
for i=1:NUM
    a1=a1+a{i}(1);
    d1=d1+d{i}(1);
    Time_train1=Time_train1+Time_train(i);
end
a1=a1/NUM;
d1=d1/NUM;
Time_train1=Time_train1/NUM;
tt1=etime(clock,t0);
Accuracy=a1;
d_feature=d1;




