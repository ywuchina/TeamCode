function eff=evaluation(pop,popsize,mm,X0,u,k)
eff=zeros(popsize,mm+2);
for i=1:popsize
    for j=1:mm
        eff(i,j)=pop(i,j);
    end
end
for i=1:popsize
    [ary,jie_sum]=kmean_res(pop(i,1:mm),X0,u,k,mm);
    eff(i,mm+1)=100*ary;
    eff(i,mm+2)=jie_sum;
end

