function Sample=Sample_grouping(kk,XX,M)
%---���в��ظ������������ǵ���ƽ������,��Ԫ����������Ӽ�����������M�������Ӽ�
Sample=cell(M,1);
 %----��ȷ����������Ŀ
 C_lable = unique(XX(:,1));
 
 %---��ͬһ������µ��������ϵ�һ��
 C_L=size(C_lable,1);
 SS=cell(C_L,1);
  for i=1:kk
     label=XX(i,1);
     index=find(C_lable==label);
     SS{index}=[SS{index};XX(i,:)];
  end
%---ÿ������£�����ѡ������
A_subnum=zeros(C_L,1);%ÿ�����Ӧ��ȡ���ٸ�����
R_a=cell(C_L,1);%ÿ���������������λ�ñ��
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