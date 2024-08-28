clear
SOL=[];
J_SOL=[];
TT=[];
Train_time=[];
A=cell(30,1);
B=cell(30,1);
for i=1:30
    [Accuracy,d_feature,tt,G,TT_train,AA_AC]=main1;
    SOL=[SOL;Accuracy];
    J_SOL=[J_SOL;d_feature];
    TT=[TT;tt];
    Train_time=[Train_time;TT_train];
    A{i}=G; 
    B{i}=AA_AC;
end
av_Ac=mean(SOL);
fc_Ac=std(SOL);
av_d=mean(J_SOL);
aTT=mean(TT)/60;%总的时间
aTrain_time=mean(Train_time)/60;%只训练的时间
 save('name30+time.mat');
 
 
 
 
