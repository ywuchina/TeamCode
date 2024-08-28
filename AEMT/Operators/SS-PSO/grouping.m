function [u,C,mm]=grouping(k,Sample,M,c,Import)

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


%---�������������н�������
C_import=c;
 [value1,index1]=sort(C_import,'descend');%���н������У�����ԭ����Ӧ��λ�÷��أ���index1=[5 4 2 1 3...]��˵������5��Ӧ����Ҫ�̶����value1,index1�ֱ���������SUֵ���Լ�SUֵ��c�ж�Ӧ��λ��
%--------����C_import��������������������ܹ���k-1������ 
  f_feature=zeros(1,k-1);%�����ʵ����������f_feature=[����7 6 3 1 4]
  f=1:k-1;%������������
 for i=1:k-1
     f_feature(i)=f(index1(i));
 end
%---------------�����Ƕ������������� 
 
 
 
 ss=value1; %��������������Ӧ��C_importֵ
 %3:ss�Ĺ�һ��?
 max1=max(value1);
 min1=min(value1);
 ss=(value1-min1)./(max1-min1);
 
 
 
 ff=f_feature;%�������������  Import(:,z)
 
 %----------------------����C_importֵ������������Ҫ�̶�������λ��,ÿһ��һ��������Ҫ����
 S_feature=zeros(M,k-1);
  for d=1:k-1
     S_feature(:,d)=Import(:,f_feature(d));
  end
  
  
  
 %-------------------------- ������Ҫ�̶Ƚ��з���
 m=1;%m��ʾ��������ĸ���
 u=cell(k,1);
 A=cell(k,1); %%A��һ��Ԫ����ÿһ��Ԫ�ر�ʾ��������Ҫ�̶����һ��������Ҫ�̶ȵĲ�ֵ
 B=cell(k,1); %%ÿһ��Ԫ�ر�ʾ��Ӧ�������һ����������س̶�
 C=cell(k,1); %%��һ��Ԫ����ÿһ��Ԫ�ر�ʾ��������Ҫ�̶�

% p=log(k-1)*abs(ss(1)-ss(k-1))/(k-1);%��Ҫ�̶Ȳ�ֵ����ֵ

%1: ----------------p
%   p=log(k-1)*norm(S_feature(:,1)-S_feature(:,k-1))/(k-1);%��Ҫ�̶Ȳ�ֵ����ֵ
% 2:---------------------------p
  p0=0;
  for l=1:k-2
      p0=p0+norm(S_feature(:,l)-S_feature(:,l+1));
  end
  p=log(k-1)*p0/(k-1);%��Ҫ�̶Ȳ�ֵ����ֵ
  %-------------------------------------
      
while(length(ff)>1)
    b=[];
    c_00=[];
%     c_1=S_feature(:,1);
          c_1=ss(1);
    d=[];
    i=2;
    z=ff(length(ff));%���һ������
    
    while(ff(i)~=z)%%%��s(i)=zʱ��˵����s�е����һ��������
        n1=norm(S_feature(:,1)-S_feature(:,i));%����C_importֵ������������Ҫ�̶�������λ��,ÿһ��һ��������Ҫ����
%         n3=S_feature(:,i);%��������������Ӧ��C_importֵ
        n3=ss(i);
        %�ж������������������
        if n1<=p
            %��ÿһ�������Ӽ��ϼ����������SU����һ���ж�������
            count=0;
            for j=1:M
                s_sample=Sample{j};
                n2=SU(s_sample(:,ff(i)+1),s_sample(:,ff(1)+1));
                yu_zhi=S_feature(j,i);
                if n2>=yu_zhi%��������������Ҫ������ȡ��j�У���i��
                    count=count+1;
                else
                    break;
                end
            end
            if count==M
                b=[b,ff(i)];ff(i)=[];ss(i)=[];S_feature(:,i)=[];c_00=[c_00,n1];d=[d,n2];c_1=[c_1,n3];
            else
                i=i+1;
            end
        else
             i=i+1;
        end
    end
   %---�����һ�����������ж�,����whileѭ����ԭ�����һ�������жϲ���
    n1=norm(S_feature(:,1)-S_feature(:,i));
    if n1<=p %%%��s�����һ�����������ж�
             %��ÿһ�������Ӽ��ϼ����������SU����һ���ж�������
            count=0;
            for j=1:M
                s_sample=Sample{j};
                n2=SU(s_sample(:,ff(i)+1),s_sample(:,ff(1)+1));
                 yu_zhi=S_feature(j,i);
                if n2>=yu_zhi%��������������Ӧ��C_importֵ
                    count=count+1;
                    break;
                end
            end
            if count==M
                b=[b,ff(i)];ff(i)=[];ss(i)=[];S_feature(:,i)=[];c_00=[c_00,n1]; d=[d,n2];c_1=[c_1,n3];
            end
    end
    b=[ff(1),b]; ff(1)=[];ss(1)=[];S_feature(:,1)=[];%%%%�ѵ�һ��������Ŀǰs������Ҫ�������������������Ѿ������������������������������b�У�Ҳ�Ƶ�b�еĵ�һλ
    u{m}=b;%%%���ȷ����b����Ԫ��u�еĵ�m��λ��
    A{m}=c_00;
    B{m}=d;
    C{m}=c_1;
    m=m+1;
end
% mm=m;%��������󹲷���mm��

if length(ff)==1%%%s����һϵ�еĻ��ֺ�����������һ�����������ٽ���������ж������֣�ֱ�ӷ���һ����
    b=[]; b=[b,ff(1)];c_1=ss(1);ff(1)=[];S_feature(:,1)=[]; u{m}=b;C{m}=c_1;
    m=m+1;
end

mm=m-1;%��������󹲷���mm��

 
 
 
 
 
 
 
 
 
 
 


