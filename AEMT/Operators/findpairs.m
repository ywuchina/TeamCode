function pair = findpairs(p1,p2,fs1,fs2,X,yuzhi)
%找到p1 p2中对应点集合
%   Detailed explanation goes here
    
%     tp=pctransform(p1,T);
%     pcshowpair(tp,p2);
    Matchmap=zeros(size(p1.Location,1),size(p2.Location,1));
    for i=1:length(fs1)
        f1=fs1{i}(:,X==1);
        f2=fs2{i}(:,X==1);
        [pairs,~]=pcmatchfeatures(f1,f2,p1,p2,"MatchThreshold",0.95,'Method','Exhaustive','RejectRatio',0.95);
        for j=1:size(pairs,1)
            x=pairs(j,1);y=pairs(j,2);
            Matchmap(x,y)=Matchmap(x,y)+1+2*i/length(fs1);
        end
    end
    %测试一下第一和第二差距小的
    [M1,idx1]=max(Matchmap,[],2);
    Matchmap(1:size(p1.Location,1),idx1)=0;
    [M2,idx2]=max(Matchmap,[],2);

    %选择第一和第二差距大的作为可靠匹配 随后对可靠匹配之间使用几何一致性进行进行筛选 选出最好的几个用于非可靠匹配
%     f1=(M1-M2)>2&M1>3;
    f1=M1>3;
    X=1:size(p1.Location,1);
    X=X(f1==1)';%X和Y是可靠匹配
    Y=idx1(f1==1);
    %用几何一致性过滤出更可靠的匹配
    while(true)
        rmap=[];%reliable map 可信赖的匹配
        p=1;
        for i=1:length(X)
            Xloc1=p1.Location(X(i),:);
            Yloc1=p2.Location(Y(i),:);
            num=0;
            for j=1:length(X)
                if(i==j)
                    continue;
                end
                Xloc2=p1.Location(X(j),:);
                Yloc2=p2.Location(Y(j),:);
                d1=sum((Xloc1-Xloc2).*(Xloc1-Xloc2));
                d1=sqrt(d1);
                d2=sum((Yloc1-Yloc2).*(Yloc1-Yloc2));
                d2=sqrt(d2);
                %之前是0.03
                if(abs(d1-d2)<yuzhi)
                    num=num+1;
                end
            end


            %这个阈值太大或太小都不好
            if(num/length(Y)>=0.5) %大部分可靠匹配都认为其满足几何一致性，则为真
                rmap(p,:)=[X(i),Y(i)];
                p=p+1;
            end
        end
        %一个也没有或者大部分已经很好了  经过实验 发现这里取1是效果最好的 
        % 必须让进行认可的匹配都是已经被认可的匹配，避免未被认可的匹配造成影响
        if(isempty(rmap)||size(rmap,1)>=length(X)*1)
            break;
        end
        X=rmap(:,1);
        Y=rmap(:,2);
    end
    pair=rmap;
    return;



    %再用可靠匹配的几何一致性去筛选不可靠匹配
    f1=(M1-M2)<=2&(M1>2);
    X=1:size(p1.Location,1);
    X=X(f1==1)';%X和Y是可靠匹配
    Y=idx1(f1==1);
    TY=idx2(f1==1);
    nrmap=[];
    while(true)
        nrmap=[];%reliable map 可信赖的匹配
        p=1;
        for i=1:length(X)
            Xloc1=p1.Location(X(i),:);
            Yloc1=p2.Location(Y(i),:);
            TYloc1=p2.Location(TY(j),:);
            num1=0;
            num2=0;
            for j=1:size(rmap,1)
                Xloc2=p1.Location(rmap(j,1),:);
                Yloc2=p2.Location(rmap(j,2),:);
                d1=sum((Xloc1-Xloc2).*(Xloc1-Xloc2));
                d1=sqrt(d1);
                d2=sum((Yloc1-Yloc2).*(Yloc1-Yloc2));
                d2=sqrt(d2);
                Td2=sum((TYloc1-Yloc2).*(TYloc1-Yloc2));
                Td2=sqrt(Td2);
%                 Td2=sum((TYloc1-TYloc2).*(TYloc1-TYloc2));
                if(abs(d1-d2)<0.03)
                    num1=num1+1;
                end
                if(abs(d1-Td2)<0.03)
                    num2=num2+1;
                end
                if num1>num2
                    if(num1/size(rmap,1)>=0.8) %大部分可靠匹配都认为其满足几何一致性，则为真
                        nrmap(p,:)=[X(i),Y(i)];
                        p=p+1;
                    end
                else
                    if(num2/size(rmap,1)>=0.8) %大部分可靠匹配都认为其满足几何一致性，则为真
                        nrmap(p,:)=[X(i),TY(i)];
                        p=p+1;
                    end
                end
            
        end
        if(isempty(nrmap)||size(nrmap,1)>=length(X)*1)
            break;
        end
        X=nrmap(:,1);
        Y=nrmap(:,2);
    end
    pairs=[rmap;nrmap];

%     pairs=[(1:size(p1.Location,1))',idx1];
%     %匹配阈值小代表匹配的数量多，也更加精确
% %     score=length(fs1)*0.8;
% %     score=2*length(fs1)-1+3;
%     score=5;
%     pairs=pairs(M1>=score,:);
%     f1=extractFPFHFeatures(p1,Radius=0.3);
%     f2=extractFPFHFeatures(p2,Radius=0.3);
%     pairs=pcmatchfeatures(f1,f2,p1,p2,"MatchThreshold",1);

%     tranp1=pctransform(p1,T);
%     pcshowMatchedFeatures(tranp1,p2,pointCloud(tranp1.Location(pairs(:,1),:)),pointCloud(p2.Location(pairs(:,2),:)));

    pair=pairs;
end