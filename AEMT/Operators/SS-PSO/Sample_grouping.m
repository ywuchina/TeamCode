function Sample=Sample_grouping(kk,XX,M)
%---进行不重复采样，并考虑到不平衡问题,用元胞存放样本子集，假设生成M个样本子集
Sample=cell(M,1);
 %----先确定出类别的数目
 C_lable = unique(XX(:,1));
 
 %---将同一个类别下的样本集合到一起
 C_L=size(C_lable,1);
 SS=cell(C_L,1);
  for i=1:kk
     label=XX(i,1);
     index=find(C_lable==label);
     SS{index}=[SS{index};XX(i,:)];
  end
%---每个类别下，均匀选择样本
A_subnum=zeros(C_L,1);%每个类别应该取多少个样本
R_a=cell(C_L,1);%每个类别中样本所在位置标号
for i=1:C_L
    a=size(SS{i},1);
    R_a{i}=randperm(a);
    A_subnum(i)=round(a/M);
end

for j=1:M-1
    for i=1:C_L
        b0=A_subnum(i);
        b1=R_a{i}(1:b0);
        Sample{j}=[Sample{j};SS{i}(b1,:)];
        R_a{i}(1:b0)=[];
    end
end

    for i=1:C_L
%         b0=A_subnum(i);
        b1=R_a{i}(1:end);
        Sample{M}=[Sample{M};SS{i}(b1,:)];
%         R_a{i}(1:b0)=[];
    end