function [matchnum,sss,scoress]=test(prm,best)
%测试特征子集效果
%   best是特征子集，prm是参数
    Maxnum=prm.Maxnum;yuzhi=prm.yuzhi;dim=prm.dim;LEN=prm.LEN;T=prm.T;p=prm.pp;F=prm.FF;data=prm.data;GrtT=prm.GrtT;GrtR=prm.GrtR;
    
    score=0;
    scoreall=0;
    scores=zeros(1,length(LEN));
    num=0;
    mapv=zeros(100,2);
    num2=0;
    zeronum=0; 
    for i=1:Maxnum
        for j=i+1:Maxnum
            num2=num2+1;
            t=[GrtR{i},GrtT{i}];
            t=[t;0 0 0 1];
            t=t';
            T1=t;
            t=[GrtR{j},GrtT{j}];
            t=[t;0 0 0 1];
            t=t';
            T2=t;
            T=round(T1/T2,10); 
            
            
            p1=p(i);p2=p(j);
            NF1=[];NF2=[];
            NF1=F{i}(:,best(1,:)==1);
            NF2=F{j}(:,best(1,:)==1);
            %特征进行匹配
            [pairs,~]=pcmatchfeatures(NF1,NF2,p1,p2,"MatchThreshold",0.95,'Method','Exhaustive','RejectRatio',0.95);
            if(isempty(pairs))
                
                numF=0;
                scoress(1,num2)=0;
                matchnum(1,num2)=0;
                continue;
            end
            np1=pointCloud(p1.Location(pairs(:,1),:));
            np2=pointCloud(p2.Location(pairs(:,2),:));
            tp1=pctransform(np1,affine3d(T));
            numF=length(pairs(:,1));
            d=zeros(1,length(tp1.Location(:,1)));
            for q=1:length(tp1.Location(:,1))
                d(q)=rms(tp1.Location(q,:)-np2.Location(q,:));
            end
            num=sum(find(d<yuzhi)~=0);
            s1=num/length(d);
            scoress(1,num2)=s1;
            matchnum(1,num2)=num;
%             if i==1&&j==5
%                 disp("第"+i+"片点云与第"+j+"片点云匹配适应度："+s1+"  匹配个数为："+size(pairs,1));
%                 fg=figure();
%                 pcshowMatchedFeatures(p1,p2,pointCloud(np1.Location(d<yuzhi,:)),pointCloud(np2.Location(d<yuzhi,:)),'PlotOptions',{'bo','bo','b-'});
%                 hold on;
%                 pcshowMatchedFeatures(p1,p2,pointCloud(np1.Location(d>yuzhi,:)),pointCloud(np2.Location(d>yuzhi,:)),'PlotOptions',{'r*','r*','r--'});
%                 set(fg,'color','w');
%                 disp("");
%             end
            
            if i==5&&j==6
                % p1=prm.originoP(i);
                % p2=prm.originoP(j);
                % tp1=pctransform(p1,affine3d(T));
                % ntp1=pctransform(np1,affine3d(T));
                % fg=figure();
                % p3=pcmerge(tp1,p2,0.01);
                % pcshow(p3);
                % set(fg,'color','w');
                % axis off;
                % % disp("第"+i+"片点云与第"+j+"片点云匹配适应度："+s1+"  匹配个数为："+size(pair,1));
                % fg=figure();
                % pcshowMatchedFeatures(tp1,p2,pointCloud(ntp1.Location(d<yuzhi,:)),pointCloud(np2.Location(d<yuzhi,:)),'PlotOptions',{'go','go','g-'});
                % 
                % hold on;
                % pcshowMatchedFeatures(tp1,p2,pointCloud(ntp1.Location(d>yuzhi,:)),pointCloud(np2.Location(d>yuzhi,:)),'PlotOptions',{'r*','r*','r--'});
                % disp("");
                % set(fg,'color','w');
                % axis off;

                
                % disp("第"+i+"片点云与第"+j+"片点云匹配适应度："+s1+"  匹配个数为："+size(pairs,1));
                % fg=figure();
                % pcshowMatchedFeatures(tp1,p2,pointCloud(tp1.Location(d<yuzhi,:)),pointCloud(np2.Location(d<yuzhi,:)),'PlotOptions',{'bo','bo','b-'});
                % hold on;
                % pcshowMatchedFeatures(tp1,p2,pointCloud(tp1.Location(d>yuzhi,:)),pointCloud(np2.Location(d>yuzhi,:)),'PlotOptions',{'r*','r*','r--'});
                % disp("");
                % set(fg,'color','w');
            end
            num=num+1;
%             disp("对第"+i+"片点云和第"+j+"片点云匹配:");
%             disp("第"+i+"片点云与第"+j+"片点云匹配适应度："+s1+"  匹配个数为："+nummm);
            score=score+s1;
%             disp("所有特征全选适应度："+s2);

%             scoreall=scoreall+s2;
%             ts=zeros(1,length(LEN));
%             for q=1:length(LEN)
%                 if q==1
%                     l=1;
%                     r=LEN(1);
%                 else
%                     l=r+1;
%                     r=r+LEN(q);
%                 end
%                 zero=zeros(1,dim);
%                 zero(l:r)=1;
%                 [s,~]=eva2(zero,p(i),p(j),F{i},F{j},affine3d(T),yuzhi);
%                 ts(q)=s;
% %                 disp("第"+q+"个描述子适应度"+q+"："+s);
%                 scores(q)=scores(q)+s;
%             end

%             if sum(s1<=ts)==0
%                 disp("测试集最好适应度："+s1);
%                 score=score+s1;
%                 num=num+1;
% %                 pcshowMatchedFeatures()
%                 disp("所有特征全选适应度："+s2);
%                 scoreall=scoreall+s2;
%                 l=LEN(1);
%                 for q=1:length(LEN)
%                     disp("适应度"+q+"："+ts(q));
%                     scores(q)=scores(q)+ts(q);
%                 end
%             end
        end
    end
    score=score/num2;
%     scoreall=scoreall/num;
%     scores=scores/num;
    sss=score;
%     disp("0的数量以及占百分比："+zeronum+"  "+zeronum/num2);
%     disp("好的数量以及占百分比："+num+"  "+num/num2);

%     disp("测试集最好子集平均适应度："+score);
%     disp("所有特征全选适应度："+scoreall);
%     for q=1:length(LEN)
%         disp("适应度"+q+"："+scores(q));
%     end
%     disp(sum(best));
%     disp(sum(best(1:33))+" "+sum(best(34:384))+" "+sum(best(385:537))+" "+sum(best(538:672)));
%     disp("")
    
%     score=0;
%     scoreall=0;
%     scores=zeros(1,length(LEN));
%     num=0;
%     mapv=zeros(100,2);
%     num2=0;
%     zeronum=0; 
%     for i=1:floor(Maxnum/4)
%         for j=floor(Maxnum/4)+1:floor(Maxnum/2)
%     for i=floor(Maxnum/3)+1:floor(Maxnum*3/4)
%         for j=floor(Maxnum*3/4)+1:floor(Maxnum/1)
%             num2=num2+1;
%             t=[GrtR{i},GrtT{i}];
%             t=[t;0 0 0 1];
%             t=t';
%             T1=t;
%             t=[GrtR{j},GrtT{j}];
%             t=[t;0 0 0 1];
%             t=t';
%             T2=t;
%             T=round(T1/T2,10); 
% 
%             [s1,~]=eva2(best,p(i),p(j),F{i},F{j},affine3d(T),yuzhi);
%             [s2,~]=eva2(ones(1,dim),p(i),p(j),F{i},F{j},affine3d(T),yuzhi);
%             num=num+1;
% %             disp("测试集最好适应度："+s1);
%             score=score+s1;
% %             disp("所有特征全选适应度："+s2);
%             scoreall=scoreall+s2;
%             ts=zeros(1,length(LEN));
%             for q=1:length(LEN)
%                 if q==1
%                     l=1;
%                     r=LEN(1);
%                 else
%                     l=r+1;
%                     r=r+LEN(q);
%                 end
%                 zero=zeros(1,dim);
%                 zero(l:r)=1;
%                 [s,~]=eva2(zero,p(i),p(j),F{i},F{j},affine3d(T),yuzhi);
%                 ts(q)=s;
% %                 disp("适应度"+q+"："+s);
%                 scores(q)=scores(q)+s;
%             end
% %             if s1>max(ts)+0.2&&max(ts)>0.4
% %                 disp(s1+" "+ts);
% %                 plant(best,p(i),p(j),F{i},F{j},affine3d(T));
% %                 title("selectbest");
% %                 zero=zeros(1,dim);
% %                 zero(384:384+153-1)=1;
% %                 plant(zero,p(i),p(j),F{i},F{j},affine3d(T));
% %                 title("initial descriptor");
% %             end
%         end
%     end
%     score=score/num;
%     scoreall=scoreall/num;
%     scores=scores/num;
%     disp("0的数量以及占百分比："+zeronum+"  "+zeronum/num2);
%     disp("好的数量以及占百分比："+num+"  "+num/num2);

%     disp("训练集最好子集平均适应度："+score);
%     disp("所有特征全选适应度："+scoreall);
%     for q=1:length(LEN)
%         disp("适应度"+q+"："+scores(q));
%     end
%     disp(sum(best));
%     disp(sum(best(1:33))+" "+sum(best(34:384))+" "+sum(best(385:537))+" "+sum(best(538:672)));
%     disp("")
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