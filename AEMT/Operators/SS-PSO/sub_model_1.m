function W_0=sub_model_1(popeff_0,Sample,k,M,mm,u,popsize)%popeff_0�Ǳ�ȫ�����������۹��ĳ�ʼ��Ⱥ��Sample�������Ӽ�Ԫ����u�������Ӽ�Ԫ����M,mm�ֱ��������Ӽ�����Ŀ�����������Ŀ
 sub_popeff=cell(M,1); %��Ԫ����������M�������Ӽ��ϵ���Ⱥ���ྫ��
 %-------��ÿ�������Ӽ��ϼ�����ྫ��
 %-------popeff_0��mm+1������ʵ��Ӧֵ��mm+2��������ģ��
 pop_0=popeff_0(:,1:mm);%��ʼ��Ⱥ

for i=1:M

    sub_popeff{i}=evaluation(pop_0,popsize,mm,Sample{i},u,k);

end
 
 
 %---------���ݳ�ʼ����������ʼȨ��//�����������
 d=popeff_0(:,mm+1);%-------popeff_0��mm+1������ʵ��Ӧֵ��
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

 
 
 
 

