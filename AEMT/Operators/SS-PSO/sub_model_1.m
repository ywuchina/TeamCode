function W_0=sub_model_1(popeff_0,Sample,k,M,mm,u,popsize)%popeff_0是被全部样本集评价过的初始种群，Sample是样本子集元胞，u是特征子集元胞，M,mm分别是样本子集的数目和特征组的数目
 sub_popeff=cell(M,1); %用元胞来保存在M个样本子集上的种群分类精度
 %-------在每个样本子集上计算分类精度
 %-------popeff_0中mm+1列是真实适应值，mm+2是特征规模。
 pop_0=popeff_0(:,1:mm);%初始种群

for i=1:M

    sub_popeff{i}=evaluation(pop_0,popsize,mm,Sample{i},u,k);

end
 
 
 %---------根据初始样本点计算初始权重//采用线性拟合
 d=popeff_0(:,mm+1);%-------popeff_0中mm+1列是真实适应值，
 C=[];
 for i=1:M
     c=sub_popeff{i};
     C=[C,c(:,mm+1)];
 end
 Aeq=[];
 beq=[];
 A=[];
 b=[];
 lb=zeros(M,1);
 ub=ones(M,1);
%  lb=0;
%  ub=1;
index=find(d==0);
if index>0
    d(index)=[];
    C(index,:)=[];
end


 
 W_0=lsqlin(C,d,A,b,Aeq,beq,lb,ub);

 
 
 
 

