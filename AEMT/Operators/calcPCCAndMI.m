function [X,time]= calcPCCAndMI(db,prm)
    tic
    Maxnum=prm.Maxnum;yuzhi=prm.yuzhi;dim=prm.dim;LEN=prm.LEN;T=prm.T;p=prm.pp;F=prm.FF;data=prm.data;GrtT=prm.GrtT;GrtR=prm.GrtR;
    pair=db.Pair;
    num=size(pair,1);%匹配的个数
    label=pair(:,5);
    PF=zeros(num,dim);
    for i=1:num
        f1=F{pair(i,1)}(pair(i,3),:);
        f2=F{pair(i,2)}(pair(i,4),:);
        PF(i,:)=rms(f1-f2,1)<yuzhi;
        PF(i,(f1==0)&(f2==0))=0;
%                 PF(i,:)=rms(f1-f2,1)<st/5;
    end
    cor=zeros(1,dim);
    for i=1:dim
        cor(1,i)=corr(PF(:,i),label);
    end
    [~,ranks]=sort(-cor);

    r=zeros(1,dim);
    for i=1:dim
        r(ranks(i))=i;
    end
    mi=zeros(1,dim);
    for i=1:dim
        mi(1,i)=MutualInfo4(PF(:,i),label);
    end
    [~,ranks]=sort(-mi);
    for i=1:dim
        if(r(ranks(i))>i)
            r(ranks(i))=i;
        end
    end
    
    [~,ranks]=sort(r);

    X=zeros(1,dim);
    X(1,ranks(1:100))=1;
    disp("perison运行时间：");
    time=toc
    
    l=1;r=33;
    s="";
    for i=1:4
        s=s+sum(X(1,l:r))+" ";
        if i~=4
            l=r+1;
            r=l+LEN(i+1)-1;
        end
    end
    disp(s);
end