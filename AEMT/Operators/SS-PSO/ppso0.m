function [LBEST,gbest]=ppso0(pop,LBEST,popsize,k,X0,gbest,mm,u)
%粒子psize,k,X)个体极值的更新－－－－－－－－－－－－－－－－
 popeff=evaluation(pop,popsize,mm,X0,u,k);
 
 for i=1:popsize
     if popeff(i,mm+1)>LBEST(i,mm+1)
         LBEST(i,1:mm+2)=popeff(i,1:mm+2);
     elseif popeff(i,mm+1)==LBEST(i,mm+1)&&popeff(i,mm+2)<LBEST(i,mm+2)
         LBEST(i,1:mm+2)=popeff(i,1:mm+2);
     end
 end

for i=1:popsize
    if LBEST(i,mm+1)>gbest(1,mm+1)
        gbest(1,1:mm+2)=LBEST(i,1:mm+2);
    elseif LBEST(i,mm+1)==gbest(1,mm+1)&&LBEST(i,mm+2)<gbest(1,mm+2)
        gbest(1,1:mm+2)=LBEST(i,1:mm+2);
    end
end
end
