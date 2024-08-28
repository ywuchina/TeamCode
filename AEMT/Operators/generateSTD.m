function r=generateSTD(prm)
    Maxnum=prm.Maxnum;yuzhi=prm.yuzhi;dim=prm.dim;LEN=prm.LEN;T=prm.T;p=prm.p;F=prm.F;data=prm.data;GrtT=prm.GrtT;GrtR=prm.GrtR;
    af=[];
    for i=1:Maxnum
        af=[af;F{i}];
    end
    st=std(af);
    r=[];
    for i=1:dim
        r(1,i)=rms(af(:,i));
    end
    
end
