function [Sample,XX,c,f,Import]=dele(Sample0,X0,k,M)

%---k�����ݼ�X��������X�ĵ�һ�д���б�ǩ������λ�ڵ�2��k-1��
%---Ԫ��Sample�з���M�������Ӽ�


%---����ÿ�������Ӽ���������Ҫ�̶Ȳ�������������
Import=zeros(M,k-1);%k-1��������������ÿ�������ĳ�����M   
 for i=1:M
     sub_sample=Sample0{i};
     max1=max(sub_sample(:,1));
     min1=min(sub_sample(:,1));
     Y=(sub_sample(:,1)-min1+0.1)./(max1-min1);
%      f_0=[];
%      c_0=[];
    for j=1:k-1
       b=SU(sub_sample(:,j+1),Y);
       Import(i,j)=b;  %c�д����Щ������Ӧ��SUֵ
    end
 end
%-------ͬһ�������Ӽ��ϣ�������������Ҫ�̶�ֵ
for i=1:M
    a1=max(Import(i,:));
    a2=min(Import(i,:));
    for j=1:k-1
        if a1==a2
            Import(i,j)=1;
        else
            Import(i,j)=(Import(i,j)-a2)/(a1-a2);
        end
    end
end


 
%---����ÿ��������Ҫ������L2��ʽ��������ÿ��������ȫ����������C-��Ҫ�̶�ֵC_import
 C_import=zeros(1,k-1);
 for z=1:k-1
     C_import(z)=norm(Import(:,z));
 end
 c_0=C_import;
 f_0=1:k-1;%������������;
 c=[];
f=[];


[value1,~]=sort(c_0,'descend');%value1���SUֵ�ҽ�������

c_c=min(0.1*value1(1),value1(floor((k-1)/log(k-1))));%log=log e

for i=1:k-1
     if (c_0(i)>c_c)
       f=[f,i];  %f�д����������
       c=[c,c_0(i)];  %c�д����Щ������Ӧ��SUֵ
     end
end

%------��Sample0������Լ��

l_f=length(f);
XX(:,1)=X0(:,1);
for m=1:l_f
    XX(:,m+1)=X0(:,f(m)+1);
end
Import0=Import;
for m=1:l_f
    Import(:,m)=Import0(:,f(m));
end 

Sample=cell(M,1);
for j=1:M
    Sample{j}(:,1)=Sample0{j}(:,1);
    for m=1:l_f
    Sample{j}(:,m+1)=Sample0{j}(:,f(m)+1);
    end
end
   

cmax=max(c);
cmin=min(c);
c=(c-cmin)./(cmax-cmin);

