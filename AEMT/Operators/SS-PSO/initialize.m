function [pop,p_c]=initialize(popsize,u,C,mm)
cv=zeros(1,mm);
for ii=1:mm
    cv(ii)=max(C{ii});
end
C_v=max(cv);
p_c=zeros(1,mm);
for ii=1:mm
    p_c(ii)=cv(ii)/C_v;
end
pop=zeros(popsize,mm);
for i=1:(popsize)
    for j=1:mm
        a=rand;
        if a<p_c(j)
            pop(i,j)=randperm(length(u{j}),1);
        else
            pop(i,j)=0;
        end
    end
end
pop(1,1:mm)=0;


