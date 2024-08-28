function ggbest=evaluationlast(gbest,mm,u,X,Y,f)    
ggbest=gbest;
 [ary,jie_sum]=lastkmean_res(ggbest(1,1:mm),X,Y,u,mm,f);
ggbest(1,mm+1)=100*ary;
ggbest(1,mm+2)=jie_sum;



