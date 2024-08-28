function [Sample,XX,c,f,Import]=dele(Sample0,X0,k,M)

%---k是数据集X的列数，X的第一列存放列标签，特征位于第2：k-1列
%---元胞Sample中放了M个样本子集


%---计算每个样本子集上特征重要程度并构成特征向量
Import=zeros(M,k-1);%k-1个特征列向量，每个特征的长度是M   
 for i=1:M
     sub_sample=Sample0{i};
     max1=max(sub_sample(:,1));
     min1=min(sub_sample(:,1));
     Y=(sub_sample(:,1)-min1+0.1)./(max1-min1);
%      f_0=[];
%      c_0=[];
    for j=1:k-1
       b=SU(sub_sample(:,j+1),Y);
       Import(i,j)=b;  %c中存放这些特征对应的SU值
    end
 end
%-------同一个样本子集上，所有特征的重要程度值
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


 
%---计算每个特征重要向量的L2范式，来衡量每个特征在全部样本集上C-重要程度值C_import
 C_import=zeros(1,k-1);
 for z=1:k-1
     C_import(z)=norm(Import(:,z));
 end
 c_0=C_import;
 f_0=1:k-1;%生成特征名称;
 c=[];
f=[];


[value1,~]=sort(c_0,'descend');%value1存放SU值且降序排列

c_c=min(0.1*value1(1),value1(floor((k-1)/log(k-1))));%log=log e

for i=1:k-1
     if (c_0(i)>c_c)
       f=[f,i];  %f中存放所有特征
       c=[c,c_0(i)];  %c中存放这些特征对应的SU值
     end
end

%------对Sample0进行列约简

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

