function [W_0,w_eff]=sub_model_0(popeff_0,Sample,k,M,mm,u,popsize)%popeff_0�Ǳ�ȫ�����������۹��ĳ�ʼ��Ⱥ��Sample�������Ӽ�Ԫ����u�������Ӽ�Ԫ����M,mm�ֱ��������Ӽ�����Ŀ�����������Ŀ
 sub_popeff=cell(M,1); %��Ԫ����������M�������Ӽ��ϵ���Ⱥ���ྫ��
 %-------��ÿ�������Ӽ��ϼ�����ྫ��
 pop_0=popeff_0(:,1:mm);%��ʼ��Ⱥ
 

% parpool 'local' 'MATLAB Parallel Cloud';
for I=1:M

    sub_popeff{I}=evaluation(pop_0,popsize,mm,Sample{I},u,k);

end
% parpool closed;




 
 
 %---------���ݳ�ʼ����������ʼȨ��//�����������
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
 %-----Ȩ�����
 
%  W=sum(W_0);
 
 %-----------���¼����Ȩ����Ӧֵ
 
 
 %------����Ȩ�ؼ���������Ӧֵ
 
 w_eff=zeros(popsize,mm+2+M);%mm+1�ż�Ȩ�������Ӧֵ
 w_eff(:,1:mm)=pop_0;
 %-------�����ӵļ�Ȩ��Ľ�����Ӧֵ,�Լ���ÿ�������Ӽ��ϵ���Ӧֵ
for i=1:popsize
     ary=0;
     sub_eff=zeros(1,M);%��������ÿ��������ÿ�������Ӽ��ϵ���Ӧֵ
     for j=1:M
         a_f=sub_popeff{j}(i,mm+1);
         sub_eff(1,j)=a_f;%������ʽ��һ��M�С�
         ary=ary+W_0(j)*a_f;
     end
     
    w_eff(i,mm+1)=ary;
    w_eff(i,mm+2)= sub_popeff{1}(i,mm+2);
    w_eff(i,mm+3:end)=sub_eff;%��mm+3�д�ŵ�һ�������Ӽ��ϵ���Ӧֵ
end


 

 
 
 
 

