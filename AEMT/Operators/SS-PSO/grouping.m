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


%---对所有特征进行降序排列
C_import=c;
 [value1,index1]=sort(C_import,'descend');%进行降序排列，并将原本对应的位置返回，若index1=[5 4 2 1 3...]，说明特征5对应的重要程度最大。value1,index1分别存放排序后的SU值，以及SU值在c中对应的位置
%--------根据C_import的排序调整特征的排序，总共有k-1个特征 
  f_feature=zeros(1,k-1);%存放真实的特征名称f_feature=[特征7 6 3 1 4]
  f=1:k-1;%生成特征名称
 for i=1:k-1
     f_feature(i)=f(index1(i));
 end
%---------------以上是对所有特征排序 
 
 
 
 ss=value1; %存放排序后特征对应的C_import值
 %3:ss的归一化?
 max1=max(value1);
 min1=min(value1);
 ss=(value1-min1)./(max1-min1);
 
 
 
 ff=f_feature;%存放排序后的特征  Import(:,z)
 
 %----------------------根据C_import值，调整特征重要程度向量的位置,每一列一个特征重要向量
 S_feature=zeros(M,k-1);
  for d=1:k-1
     S_feature(:,d)=Import(:,f_feature(d));
  end
  
  
  
 %-------------------------- 根据重要程度进行分组
 m=1;%m表示子特征组的个数
 u=cell(k,1);
 A=cell(k,1); %%A是一个元胞，每一个元素表示特征的重要程度与第一个特征重要程度的差值
 B=cell(k,1); %%每一个元素表示对应特征与第一个特征间相关程度
 C=cell(k,1); %%是一个元胞，每一个元素表示特征的重要程度

% p=log(k-1)*abs(ss(1)-ss(k-1))/(k-1);%重要程度差值的阈值

%1: ----------------p
%   p=log(k-1)*norm(S_feature(:,1)-S_feature(:,k-1))/(k-1);%重要程度差值的阈值
% 2:---------------------------p
  p0=0;
  for l=1:k-2
      p0=p0+norm(S_feature(:,l)-S_feature(:,l+1));
  end
  p=log(k-1)*p0/(k-1);%重要程度差值的阈值
  %-------------------------------------
      
while(length(ff)>1)
    b=[];
    c_00=[];
%     c_1=S_feature(:,1);
          c_1=ss(1);
    d=[];
    i=2;
    z=ff(length(ff));%最后一个特征
    
    while(ff(i)~=z)%%%当s(i)=z时，说明到s中的最后一个特征了
        n1=norm(S_feature(:,1)-S_feature(:,i));%根据C_import值，调整特征重要程度向量的位置,每一列一个特征重要向量
%         n3=S_feature(:,i);%存放排序后特征对应的C_import值
        n3=ss(i);
        %判断两个特征不是弱相关
        if n1<=p
            %在每一个样本子集上计算特征间的SU，进一步判断冗余性
            count=0;
            for j=1:M
                s_sample=Sample{j};
                n2=SU(s_sample(:,ff(i)+1),s_sample(:,ff(1)+1));
                yu_zhi=S_feature(j,i);
                if n2>=yu_zhi%存放排序后特征重要向量，取第j行，第i列
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
   %---对最后一个特征进行判断,由于while循环的原因。最后一个特征判断不到
    n1=norm(S_feature(:,1)-S_feature(:,i));
    if n1<=p %%%对s中最后一个特征进行判断
             %在每一个样本子集上计算特征间的SU，进一步判断冗余性
            count=0;
            for j=1:M
                s_sample=Sample{j};
                n2=SU(s_sample(:,ff(i)+1),s_sample(:,ff(1)+1));
                 yu_zhi=S_feature(j,i);
                if n2>=yu_zhi%存放排序后特征对应的C_import值
                    count=count+1;
                    break;
                end
            end
            if count==M
                b=[b,ff(i)];ff(i)=[];ss(i)=[];S_feature(:,i)=[];c_00=[c_00,n1]; d=[d,n2];c_1=[c_1,n3];
            end
    end
    b=[ff(1),b]; ff(1)=[];ss(1)=[];S_feature(:,1)=[];%%%%把第一个特征（目前s中最重要的特征，上述操作，已经把相关性满足条件的所有特征放在b中）也移到b中的第一位
    u{m}=b;%%%最后确定的b放在元胞u中的第m个位置
    A{m}=c_00;
    B{m}=d;
    C{m}=c_1;
    m=m+1;
end
% mm=m;%分组结束后共分了mm组

if length(ff)==1%%%s经过一系列的划分后，如果最后还余下一个特征，不再进行相关性判断来划分，直接放在一个组
    b=[]; b=[b,ff(1)];c_1=ss(1);ff(1)=[];S_feature(:,1)=[]; u{m}=b;C{m}=c_1;
    m=m+1;
end

mm=m-1;%分组结束后共分了mm组

 
 
 
 
 
 
 
 
 
 
 


