function init(prm)
Maxnum=prm.Maxnum;T=prm.T;
pp=prm.pp;
FF=prm.FF;
adL=[];
for i=1:Maxnum
%     TT=T{1}.T/T{i}.T;
%     TTT=round(TT,10);
    tdp=pctransform(pp(i),affine3d(T{i}.T));
    adL=[adL;tdp.Location];
end
ad=pointCloud(adL);
ad=pcdownsample(ad,'nonuniformGridSample',150);
% for i=1:1
%     [dp(i),idx]=pcdownsample(pp(i),'nonuniformGridSample',6);
%     df{i}=FF{i}(idx,:);
% end
for i=1:Maxnum
    TT=T{1}.T/T{i}.T;
    TTT=round(TT,10);
    tdp=pctransform(ad,affine3d(TTT));
    idx=knnsearch(pp(i).Location,tdp.Location);
    
    dp(i)=pointCloud(pp(i).Location(idx,:));
    df{i}=FF{i}(idx,:);
end
save("Data\PC\ddp",'df','dp');
% generateInfo(pp,FF,T,yuzhi,LEN);
end

function generateInfo(p,F,T,yuzhi,LEN)
% 生成匹配特征矩阵和匹配标签 PF label
    stdF=zeros(1,size(F{1},2));%每个特征的标准差
    af=[];
    for i=1:length(F)
        af=[af;F{i}];
    end
    stdF=std(af,0,1);
    Maxnum=20;
    label=[];
    pair=[];
    SUM=LEN;
    for i=2:length(LEN)
        SUM(i)=SUM(i-1)+LEN(i);
    end
    Ix=zeros(sum(LEN),1);
    Ixx=zeros(length(LEN),1);
    for I=1:length(LEN)
        l=0;r=SUM(I);
        PF=[];
        tlabel=[];

        if I==1
            l=1;
        else
            l=SUM(I-1)+1;
        end
        for i=1:floor(Maxnum/2)
            for j=i+1:floor(Maxnum/2)
                p1=p(i);p2=p(j);
                f1=F{i}(:,l:r);f2=F{j}(:,l:r);
%                 disp("多的：")
%                 tic;
                [pairs,~]=pcmatchfeatures(f1,f2,p1,p2,"MatchThreshold",0.95,'Method','Exhaustive','RejectRatio',0.95);
%                 toc;
                tpair=zeros(size(pairs,1),4);
                tpair(1:size(pairs,1),1)=i;
                tpair(1:size(pairs,1),2)=j;
                tpair(1:size(pairs,1),3:4)=pairs;
                pair=[pair;tpair];
                np1=pointCloud(p1.Location(pairs(:,1),:));
                np2=pointCloud(p2.Location(pairs(:,2),:));
                TT=round(T{i}.T/T{j}.T,10);
                tp1=pctransform(np1,affine3d(TT));
                d=zeros(size(pairs,1),1);
                d=rms(tp1.Location-np2.Location,2); %第q对匹配点对的真实欧氏距离

                if isempty(pairs)
                    continue;
                end
                for q=1:size(pairs,1)
                    dd=[];
                    ff1=f1(pairs(q,1),:);
                    ff2=f2(pairs(q,2),:);
                    for z=1:length(ff1)
                        if(ff1(z)==0||ff2(z)==0)
                            dd(q,z)=1;
                        else
                            dd(q,z)=rms(ff1(z)-ff2(z));%两维特征在这对匹配上的距离距离
                        end
                        
                    end
                    FF=[dd(q,:)<yuzhi];
                    
                    PF=[PF;FF];
                    
                    

                end
                tlabel=[tlabel;d<yuzhi];%匹配正确的数量
                
            end
        end
        disp(sum(tlabel)+" "+length(tlabel)+" "+sum(tlabel)/length(tlabel));
        label=[label;tlabel];
        for i=l:r
%             Ix(i)=MutualInfo(dd(:,i-l+1),d);
%             Ix(i)=MutualInfo2(PF(:,i-l+1),tlabel);
            Ix(i)=MutualInfo4(PF(:,i-l+1),tlabel);
%             Ix(i)=(sum(PF(:,i-l+1)==1&tlabel==1)+0.5*sum(PF(:,i-l+1)==0&tlabel==0))/(1.5*length(tlabel));
%             Ix(i)=sum(PF(:,i-l+1)==tlabel)/length(tlabel);
        end
        
    end
    Ix(stdF<yuzhi)=0;
    save("Data\PC\PF&Label",'label','pair');
    save("Data\PC\I.mat",'Ix');
end