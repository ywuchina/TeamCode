function [W_0,w_eff]=sub_model_0(popeff_0,Sample,k,M,mm,u,popsize)%popeff_0是被全部样本集评价过的初始种群，Sample是样本子集元胞，u是特征子集元胞，M,mm分别是样本子集的数目和特征组的数目
 sub_popeff=cell(M,1); %用元胞来保存在M个样本子集上的种群分类精度
 %-------在每个样本子集上计算分类精度
 pop_0=popeff_0(:,1:mm);%初始种群
 

% parpool 'local' 'MATLAB Parallel Cloud';
for I=1:M

    sub_popeff{I}=evaluation(pop_0,popsize,mm,Sample{I},u,k);

end
% parpool closed;




 
 
 %---------根据初始样本点计算初始权重//采用线性拟合
 d=popeff_0(:,mm+1);
 C=zeros(popsize,M);
 for i=1:M
     c=sub_popeff{i};
     C(:,i)=c(:,mm+1);
 end

 lb=zeros(M,1);
 ub=ones(M,1);
%  lb=0;
%  ub=1;
index=find(d==0);
if index>0
    d(index)=[];
    C(index,:)=[];
end

 Aeq=[];
 beq=[];
 A=[];
 b=[];



 W_0=lsqlin(C,d,A,b,Aeq,beq,lb,ub);
%  W_0=ones(M,1);
 %-----权重求和
 
%  W=sum(W_0);
 
 %-----------重新计算加权后适应值
 
 
 %------利用权重集成最后的适应值
 
 w_eff=zeros(popsize,mm+2+M);%mm+1放加权后近似适应值
 w_eff(:,1:mm)=pop_0;
 %-------求粒子的加权后的近似适应值,以及在每个样本子集上的适应值
for i=1:popsize
     ary=0;
     sub_eff=zeros(1,M);%用来保存每个个体在每个样本子集上的适应值
     for j=1:M
         a_f=sub_popeff{j}(i,mm+1);
         sub_eff(1,j)=a_f;%最终形式是一行M列。
         ary=ary+W_0(j)*a_f;
     end
     
    w_eff(i,mm+1)=ary;
    w_eff(i,mm+2)= sub_popeff{1}(i,mm+2);
    w_eff(i,mm+3:end)=sub_eff;%从mm+3列存放第一个样本子集上的适应值
end


 

 
 
 
 

