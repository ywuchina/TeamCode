function [task,pool,task_size] = clusting_task(X,n_feature,fea_rank,pool)
mul=[];
task=[];
task_size=zeros(10,1);
mul(1:10,1)=fea_rank(1:10,1);%记录互信息，第一列为聚类中心
max_fea=ceil(n_feature/10);
for i=1:10
    pos=mul(i,1);
   for j = 1:n_feature
        mul(i,j+1) = MItest(X(:,j),X(:,pos));
    end 
end
task_pos=[];
for i=2:(n_feature+1)
    [value,pos]=max(mul(:,i));
    if task_size(pos,1)>=max_fea
        aa=1;
        mul(pos,i:end)=0;
    end
    task_pos(i-1)=pos;
    task_size(pos,1)=task_size(pos,1)+1;
end
for i=1:10
    for j=1:n_feature
        if task_pos(j)==i
            task(i,j)=1;
%             task_size(i,1)=task_size(i,1)+1;
        else
            task(i,j)=0;
        end
    end
end
% task(10,:)=ones();
% task_size(10,1)=n_feature;
aa=1;
end