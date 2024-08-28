 function  [gbest,u,mm,f,tt_train,AC]=main2(X_train)
t1=clock;
X0=X_train;
Num_hang=size(X0,1);
Num_lie=size(X0,2);

big_Num=0.5;
small_Num=0.1;
diff=big_Num-small_Num;
aa=0.5*rand*diff+small_Num;
Num_subsample=Num_hang*aa;%每个样本子集的规模
M=ceil(Num_hang/(Num_subsample));%向上取整
%---------------样本集合划分
Sample=Sample_grouping(Num_hang,X0,M);
%-------------去除不相关特征
[Sample,XX,c,f,Import]=dele(Sample,X0,Num_lie,M);
[kk,k]=size(XX);
[u,C,mm]=grouping(k,Sample,M,c,Import);
%----------------------------------------------代理辅助的进化优化
popsize=50;   
tt=100;
%-----------------------------------初始化
[pop,p_c]=initialize(popsize,u,C,mm);
 popeff=evaluation(pop,popsize,mm,XX,u,k); 

[W_0,w_eff]=sub_model_0(popeff,Sample,k,M,mm,u,popsize);%w_eff是个体群popeff，只是mm+1列放加权后近似适应值


%---------初始极值点的更新，
LBEST_zhen=popeff;%LBEST粒子个体极值，mm+1列真实值
[~,va2]=sort(LBEST_zhen(:,mm+1));
gbest=LBEST_zhen(va2(popsize),1:mm+2);%gbest的mm+1列放真实值
LBEST=w_eff;%mm+1列加权后近似值%-------为了保证在同一个权重情况下进行个体的优劣的比较，在进化过程中第一代需要带着加权后适应值的尾巴
            %从mm+3列存放第一个样本子集上的适应值
gbest(mm+3)=LBEST(va2(popsize),mm+1);%gbest的mm+3列放加权适应值

%---------初始样本点存入到AD中
A0=zeros(popsize,1);
AD=[pop,A0,popeff(:,mm+2),popeff(:,mm+1)];%此刻popeff(:,mm+1)是真实值，为了和后面mm+3是真实值保持一致，mm+1是加权后适应值，首先将mm+1全部置为0
%-------专门用来存放gbest的真实适应值从而方便判断样本是否需要重新划分
%-------AD中1:mm列是特征，mm+1是加权后适应值，mm+2是特征规模，mm+3是真实适应值，
% G_feal=gbest(mm+1);
%--------------------------------------迭代过程
t=1;
TT=tt*0.75;
AC=zeros(tt,mm+3);   
W=W_0;%首先令权重为初始值
G_feal=gbest(mm+1);%存放真实适应值
% AG=G_feal;
% AM=[];
flag=1;
    while(t<=TT && M>1)
        %------种群位置更新
         pop=up_ppso(pop,gbest,LBEST,mm,popsize,u,p_c);
        %-------------------------------------
        
        %------根据加权后近似适应值，更新全局极值点和个体极值点，更新AD
        [LBEST,gbest,AD,W,Sample,M,pop,flag]=ppso1(pop,LBEST,popsize,k,XX,gbest,mm,u,W,Sample,M,AD,flag); 
        %-------------------------------------
        %------------------------------------接下来利用当前种群的一些结果进行判断是否更新权重，还是重新划分样本后+更新权重
        
        %------判断是否需要重新划分样本，若需要，重新划分后，重新更新当前环境下的初始种群pop，并同时更新权重
        %------最后更新划分时刻的全局极值点和个体极值点
        if G_feal==gbest(mm+1)
           flag=flag+1;G_feal=gbest(mm+1);
        else
           flag=1;G_feal=gbest(mm+1);%如果在某一代出现不相等，则重置flag
        end     
        %--------若特征子集是空集，直接跳出程序
        if gbest(1,mm+1)==0&&gbest(1,mm+2)==0
            break
        end   
       %-------------------------------------
       AC(t,:)=gbest;
%        AM=[AM;M];
       t=t+1;
%        AW{t}=W;
    end
    
    while((t>TT && t<=tt) || (M==1 && t<=tt))
         pop=up_ppso(pop,gbest,LBEST,mm,popsize,u,p_c);
        [LBEST,gbest]=ppso0(pop,LBEST,popsize,k,XX,gbest,mm,u); %个体和全局极值
        if gbest(1,mm+1)==0&&gbest(1,mm+2)==0
            break
        end  
      AC(t,:)=gbest;
     t=t+1;
    end   
tt_train=etime(clock,t1);
