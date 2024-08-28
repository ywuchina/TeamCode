function R=MutualInfo4(X,Y)
% 评估两个向量之间互信息
%
%     R1=getR1(X,Y);
    if X==Y
        R=1;
        return;
    end
%     R=getR2(X,Y)+getR2(Y,X);
    Hx=getH(X);
    Hy=getH(Y);
    Hxy=getHxy(X,Y);
    R2=(Hx+Hy-Hxy)/getH(X)+(Hx+Hy-Hxy)/getH(Y);
    R=R2/2;
    if isnan(R)
        R=0;
    end
end
function R=getR2(X,Y)
    R=0;
    for x=0:1
        px=getP(X==x);
        if px==0
            continue;
        end
        for y=0:1
            py=getP(Y==y);
            if py==0
                continue;
            end
            pxy=getP(X==x&Y==y);
            R=R+pxy*log2(pxy/(px*py));
        end
    end
    R=R/getH(X);
%     Hx=getH(X);
%     Hy=getH(Y);
%     Hxy=getHxy(X,Y);
%     Ixy=Hx+Hy-Hxy;
%     R=Ixy/(Hx+Hy);
%     if isnan(R)
%         R=0;
%     end
end
function Hxy=getHxy(X,Y)
    pxy(1)=getP(X.*Y);
    pxy(2)=getP(X.*(1-Y));
    pxy(3)=getP((1-X).*Y);
    pxy(4)=getP((1-X).*(1-Y));
    for i=1:4
        v(i)=pxy(i)*log2(pxy(i));
        if isnan(v(i))
            v(i)=0;
        end
    end
    Hxy=-sum(v);
end
function H=getH(X)
    px=getP(X);
    py=getP(1-X);
    H=0;
    if px~=0 
        H=-px*log2(px);
    end
    if py~=0
        H=H-py*log2(py);
    end
    
end
function p=getP(X)
    p=sum(X)/length(X);
end