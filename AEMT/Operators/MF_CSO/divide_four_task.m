function [task,task_size,pool] = divide_four_task(X,n_feature,idx,weight,weight1,weight2,pool)
   
   [kneepoint,~]=find_knee(weight);%relif-f
   [kneepoint1,~]=find_knee(weight1);%tv
   [kneepoint2,~]=find_knee(weight2);%pcc
   task_size=zeros(4,1);
     task=[];
     for i=1:n_feature
         task(4,i)=1;
         if weight(i)>kneepoint%relif-f
             task(1,i)=1;
             task_size(1,1)=task_size(1,1)+1;
         else
             task(1,i)=0;
         end
         if weight1(i)>kneepoint1%tv
             task(2,i)=1;
             task_size(2,1)=task_size(2,1)+1;
         else
             task(2,i)=0;
         end
         if weight2(i)>kneepoint2%pcc
             task(3,i)=1;
             task_size(3,1)=task_size(3,1)+1;
         else
             task(3,i)=0;
         end
          task_size(4,1)=task_size(4,1)+1;
     end
     aa=1;
end

function [kneepoint,task_size] = find_knee(weight)
  for i=1:size(weight,2)
         task_size(i,1)=i;
         task_size(i,2)=weight(1,i);
   end
     relieff_task=sortrows(task_size,2);
     task=[];
task_size=zeros(2,1);
%      plot(task(:,2));
     x1 = 1;
     y1 = relieff_task(1,2);
     x2 = size(relieff_task,1);
     y2 = relieff_task(x2,2);
     k = (y1-y2)/(x1-x2);
     b = y1-k*x1;
     d = zeros(size(relieff_task,1),1);
     for i = 1:size(d,1)
         d(i) = abs(k*i+b-relieff_task(i,2))/sqrt(k*k+1);
     end
     [~,kneepoint_idx] = max(d);
     kneepoint=relieff_task(kneepoint_idx,2);
end
