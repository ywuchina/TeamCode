function pair = generatePair(X,prm)
%生成匹配pair
    Maxnum=prm.Maxnum;yuzhi=prm.yuzhi;dim=prm.dim;LEN=prm.LEN;T=prm.T;p=prm.p;F=prm.F;data=prm.data;GrtT=prm.GrtT;GrtR=prm.GrtR;
    pair=[];
    for q=1:size(X,1)
        for i=1:floor(Maxnum/4)
            for j=floor(Maxnum/4)+1:floor(Maxnum/2)
                p1=p(i);p2=p(j);
                f1=F{i}(:,X(q,:)==1);f2=F{j}(:,X(q,:)==1);
                %RejectRatio太大了匹配对的数量就不足，效果不好，表现不出标签的特性， 太小了也不行，匹配数量太多，太慢
                [pairs,~]=pcmatchfeatures(f1,f2,p1,p2,"MatchThreshold",0.95,'RejectRatio',0.90);

                if isempty(pairs)
                    continue;
                end

                tpair=zeros(size(pairs,1),5);
                tpair(:,1)=i;
                tpair(:,2)=j;
                tpair(:,3:4)=pairs;

                np1=pointCloud(p1.Location(pairs(:,1),:));
                np2=pointCloud(p2.Location(pairs(:,2),:));
                TT=round(T{i}.T/T{j}.T,10);
                tp1=pctransform(np1,affine3d(TT));
                d=rms(tp1.Location-np2.Location,2)';
                
                tpair(:,5)=(d<yuzhi)';

                pair=[pair;tpair];
            end
        end
    end
    pair=unique(pair,'rows');
end