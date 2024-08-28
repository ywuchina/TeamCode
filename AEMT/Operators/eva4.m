function fits = eva4(X,prm)
    Maxnum=prm.Maxnum;yuzhi=prm.yuzhi;dim=prm.dim;LEN=prm.LEN;T=prm.T;p=prm.p;F=prm.F;data=prm.data;GrtT=prm.GrtT;GrtR=prm.GrtR;
    pop=size(X,1);
    fits=zeros(pop,1);
    for q=1:pop
        num=0;
        for i=1:floor(Maxnum/2)
            for j=(i+1):floor(Maxnum/2)
                num=num+1;
                TT=T{i}.T/T{j}.T;
                TT=round(TT,10);
                fits(q)=fits(q)+eva2(X(q,:),p(i),p(j),F{i},F{j},affine3d(TT),yuzhi,prm.st);
            end
        end
        fits(q)=fits(q)/num;
    end
end
