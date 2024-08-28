function [task,pool] = divide_task(X,n_feature,fea_rank,pool)
task=[];  
k=1;
  while(pool(fea_rank(k,1))~=1)
     k=k+1;
  end
  pos=fea_rank(k,1);
  pool(fea_rank(k,1))=0;
%   task=[task;pos];
  j=1;
    for i = 11:n_feature
        if (pool(i)==0)||(i<11)
            mul(i)=0;
        else
             mul(i) = MItest(X(:,i),X(:,pos));
             V(j)=mul(i);
             j=j+1;
        end
    end
%      s=sum(pool);
%     if sum(pool)<=1
%  %        p=zeros(n_feature);
%         p=0;
%     else
%         p=[(mul-min(V))/(max(mul)-min(V))];
%           p=min(V)/max(mul);
%     end
% p=sum(mul)/sum(pool);
 p=[(mul-min(V))/(max(mul)-min(V))];
    
if sum(pool)<20
     for i=1:n_feature
         if (pool(i)==1)||(i==pos)
             task=[task;1];
             pool(i)=0;
         else
             task=[task;0];
         end
     end
else
     for i=1:n_feature
         if ((rand<p(i))&&(pool(i)==1))||(i==pos)
             task=[task;1];
             pool(i)=0;
         else
             task=[task;0];
         end
    end
end


%     task_num=sort(task);
    
     aa=1;
end

