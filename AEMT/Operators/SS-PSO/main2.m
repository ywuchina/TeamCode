 function  [gbest,u,mm,f,tt_train,AC]=main2(X_train)
t1=clock;
X0=X_train;
Num_hang=size(X0,1);
Num_lie=size(X0,2);

big_Num=0.5;
small_Num=0.1;
diff=big_Num-small_Num;
aa=0.5*rand*diff+small_Num;
Num_subsample=Num_hang*aa;%ÿ�������Ӽ��Ĺ�ģ
M=ceil(Num_hang/(Num_subsample));%����ȡ��
%---------------�������ϻ���
Sample=Sample_grouping(Num_hang,X0,M);
%-------------ȥ�����������
[Sample,XX,c,f,Import]=dele(Sample,X0,Num_lie,M);
[kk,k]=size(XX);
[u,C,mm]=grouping(k,Sample,M,c,Import);
%----------------------------------------------�������Ľ����Ż�
popsize=50;   
tt=100;
%-----------------------------------��ʼ��
[pop,p_c]=initialize(popsize,u,C,mm);
 popeff=evaluation(pop,popsize,mm,XX,u,k); 

[W_0,w_eff]=sub_model_0(popeff,Sample,k,M,mm,u,popsize);%w_eff�Ǹ���Ⱥpopeff��ֻ��mm+1�зż�Ȩ�������Ӧֵ


%---------��ʼ��ֵ��ĸ��£�
LBEST_zhen=popeff;%LBEST���Ӹ��弫ֵ��mm+1����ʵֵ
[~,va2]=sort(LBEST_zhen(:,mm+1));
gbest=LBEST_zhen(va2(popsize),1:mm+2);%gbest��mm+1�з���ʵֵ
LBEST=w_eff;%mm+1�м�Ȩ�����ֵ%-------Ϊ�˱�֤��ͬһ��Ȩ������½��и�������ӵıȽϣ��ڽ��������е�һ����Ҫ���ż�Ȩ����Ӧֵ��β��
            %��mm+3�д�ŵ�һ�������Ӽ��ϵ���Ӧֵ
gbest(mm+3)=LBEST(va2(popsize),mm+1);%gbest��mm+3�зż�Ȩ��Ӧֵ

%---------��ʼ��������뵽AD��
A0=zeros(popsize,1);
AD=[pop,A0,popeff(:,mm+2),popeff(:,mm+1)];%�˿�popeff(:,mm+1)����ʵֵ��Ϊ�˺ͺ���mm+3����ʵֵ����һ�£�mm+1�Ǽ�Ȩ����Ӧֵ�����Ƚ�mm+1ȫ����Ϊ0
%-------ר���������gbest����ʵ��Ӧֵ�Ӷ������ж������Ƿ���Ҫ���»���
%-------AD��1:mm����������mm+1�Ǽ�Ȩ����Ӧֵ��mm+2��������ģ��mm+3����ʵ��Ӧֵ��
% G_feal=gbest(mm+1);
%--------------------------------------��������
t=1;
TT=tt*0.75;
AC=zeros(tt,mm+3);   
W=W_0;%������Ȩ��Ϊ��ʼֵ
G_feal=gbest(mm+1);%�����ʵ��Ӧֵ
% AG=G_feal;
% AM=[];
flag=1;
    while(t<=TT && M>1)
        %------��Ⱥλ�ø���
         pop=up_ppso(pop,gbest,LBEST,mm,popsize,u,p_c);
        %-------------------------------------
        
        %------���ݼ�Ȩ�������Ӧֵ������ȫ�ּ�ֵ��͸��弫ֵ�㣬����AD
        [LBEST,gbest,AD,W,Sample,M,pop,flag]=ppso1(pop,LBEST,popsize,k,XX,gbest,mm,u,W,Sample,M,AD,flag); 
        %-------------------------------------
        %------------------------------------���������õ�ǰ��Ⱥ��һЩ��������ж��Ƿ����Ȩ�أ��������»���������+����Ȩ��
        
        %------�ж��Ƿ���Ҫ���»�������������Ҫ�����»��ֺ����¸��µ�ǰ�����µĳ�ʼ��Ⱥpop����ͬʱ����Ȩ��
        %------�����»���ʱ�̵�ȫ�ּ�ֵ��͸��弫ֵ��
        if G_feal==gbest(mm+1)
           flag=flag+1;G_feal=gbest(mm+1);
        else
           flag=1;G_feal=gbest(mm+1);%�����ĳһ�����ֲ���ȣ�������flag
        end     
        %--------�������Ӽ��ǿռ���ֱ����������
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
        [LBEST,gbest]=ppso0(pop,LBEST,popsize,k,XX,gbest,mm,u); %�����ȫ�ּ�ֵ
        if gbest(1,mm+1)==0&&gbest(1,mm+2)==0
            break
        end  
      AC(t,:)=gbest;
     t=t+1;
    end   
tt_train=etime(clock,t1);
