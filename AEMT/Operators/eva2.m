function [score,numF]=eva2(X,src,tar,F1,F2,T,yuzhi,st)
%对src 和 tar进行使用特征进行匹配，并不是一一对应关系
%F1 F2是用来特征匹配的 src tar是用来判断匹配的,通过将匹配点进行转换后看看是否小于阈值    
    %先将特征值拼接好
    NF1=[];NF2=[];
    X(X>0.5)=1;
    X(X<=0.5)=0;
    if sum(X)==0
        score=0;
        numF=0;
        return;
    end

    NF1=F1(:,X(1,:)==1);
    NF2=F2(:,X(1,:)==1);
    st=st(:,X(1,:)==1);
%     for i=1:size(X,2)
% %         if X(1,i)>0.8
% %             NF1=[NF1,F1(:,i)];
% %             NF2=[NF2,F2(:,i)];
% %         elseif X(1,i)>0.2
% %             NF1=[NF1,F1(:,i)*X(1,i)];
% %             NF2=[NF2,F2(:,i)*X(1,i)];
% %         else
% % 
% %         end
%         NF1=[NF1,F1(:,i)*X(1,i)];
%         NF2=[NF2,F2(:,i)*X(1,i)];
%     end
    
    %特征进行匹配
    [pairs,~]=pcmatchfeatures(NF1,NF2,src,tar,"MatchThreshold",0.95,'Method','Exhaustive','RejectRatio',0.95);
    if(isempty(pairs))
        score=0;
        numF=0;
        return;
    end
    np1=pointCloud(src.Location(pairs(:,1),:));
    np2=pointCloud(tar.Location(pairs(:,2),:));
    tp1=pctransform(np1,T);
    numF=length(pairs(:,1));
    
    for i=1:length(tp1.Location(:,1))
        d(i)=rms(tp1.Location(i,:)-np2.Location(i,:));
    end
    for i=1:length(tp1.Location(:,1))
        fd(i)=rms((NF1(pairs(i,1),:)-NF2(pairs(i,2),:)));
    end
    d=0.95*d+0.05*fd;
    num=sum(find(d<yuzhi)~=0);
    score=num/length(d);
end
function pairs=pcmatchfeatures2(f1,f2)
    d=zeros(length(f1),length(f2));
    t=1;
    pairs=[];
    for i=1:length(f1)
        for j=1:length(f2)
            d(i,j)=rms(f1(i,:)-f2(j,:));
        end
        [md,mdi]=min(d(i,:));
        if md<0.12/2
            pairs(t,1)=i;
            pairs(t,2)=mdi;
            t=t+1;
        end
    end
    
end