function [win,lose]=compare(pop,p_new,run_time)
%COMPARE 此处显示有关此函数的摘要
%   此处显示详细说明
  a=pop(p_new(run_time));
  b=pop(p_new(run_time+1));
%  if dominates(a.cost(),b.cost())
 if a.cost<b.cost
     win=p_new(run_time);
     lose=p_new(run_time+1);
 else
     win=p_new(run_time+1);
     lose=p_new(run_time);
 end 
end

