function pop=up_ppso(pop,gbest,LBEST,mm,popsize,u,p_c)
%产生新的粒子－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－
for i=1:popsize
    dlt=0;
    for ii=1:mm
        if LBEST(i,ii)==gbest(1,ii)
            dlt=dlt+1;
        end
    end
    p_m=(dlt/mm)/mm;
     fly=0;
    c1=0.5;
    c2=1-c1;
    for j=1:mm
        if(rand<0.5)
            a=ceil(c1*LBEST(i,j)+c2*gbest(1,j));
            b=abs(LBEST(i,j)-gbest(1,j));
            pop(i,j)= round( a + b*randn);
        else
            pop(i,j)=LBEST(i,j);
        end
        if rand<p_m
            pop(i,j)=randperm(length(u{j})+1,1)-1;
            fly=1;
        end
    end
    if  dlt/mm==1
        for j=1:mm
            if rand>0.5
                %         sss=ceil(mm*rand);
                pop(i,j)=0;
            end
        end
    end
    %%%----------------------------------边界
    for j=1:mm
        if pop(i,j)<0
            pop(i,j)=0;
        elseif pop(i,j)>length(u{j})
            pop(i,j)=length(u{j});
        end
    end
    
end

