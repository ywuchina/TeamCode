function [matchnum,scores]=testMultiscale(prm,xunhuannum,X)
%测试多尺度效果
%   X是特征子集
    Maxnum=prm.Maxnum;yuzhi=prm.yuzhi;dim=prm.dim;LEN=prm.LEN;T=prm.T;p=prm.pp;F=prm.FF;data=prm.data;GrtT=prm.GrtT;GrtR=prm.GrtR;
    r=7;
    score=0;
    scores=[];
    matchnum=[];

    pnum=0;
    for i=1:Maxnum
        for j=i+1:Maxnum
            pnum=pnum+1;
        end
    end
    scores=zeros(1,pnum);
    matchnum=zeros(1,pnum);
    af=readAllRadiusF(r);
    
    Ers=[];
    Ets=[];
    for z=1:xunhuannum
        num2=0;
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
    
                %找到匹配对
                pair=findpairs(p1,p2,F{i},F{j},yuzhi);
                
                if isempty(pair)
    %                 disp("第"+i+"片点云与第"+j+"片点云匹配适应度："+0+"  匹配个数为："+0);
                    
                    continue;
                end
    
                np1=pointCloud(p1.Location(pair(:,1),:));
                np2=pointCloud(p2.Location(pair(:,2),:));
                tp1=pctransform(np1,affine3d(T));
                
                d=[];
                for q=1:length(tp1.Location(:,1))
                    d(q)=rms(tp1.Location(q,:)-np2.Location(q,:));
                end
                num=sum(find(d<yuzhi)~=0);
                s1=num/length(d);
                scores(1,num2)=scores(1,num2)+s1;
                matchnum(1,num2)=matchnum(1,num2)+num;
                score=score+s1;
%                 tpp=pctransform(p1,affine3d(T));
    %             disp("第"+i+"片点云与第"+j+"片点云匹配适应度："+s1+"  匹配个数为："+size(pair,1));
                if i==5&&j==6
                    % randomInts=randperm(size(pair,1));
                    % sampleIdx = randomInts(1:3);
                    % TT=estimateTransform(np1.Location(sampleIdx,:)',np2.Location(sampleIdx,:)');
                    % TT=TT';
                    p1=p(i);p2=p(j);

                    % p1=prm.originoP(i);
                    % p2=prm.originoP(j);
                    tp1=pctransform(p1,affine3d(T));
                    ntp1=pctransform(np1,affine3d(T));
                    p3=pcmerge(tp1,p2,0.01);
                    % pcshow(p3);
                    fg=figure();
                    pcshow(p1);
                    set(fg,'color','w');
                    axis off;
                    fg=figure();
                    pcshow(p2);
                    set(fg,'color','w');
                    axis off;
                    fg=figure();
                    pcshow(p3);
                    set(fg,'color','w');
                    axis off;
                    % disp("第"+i+"片点云与第"+j+"片点云匹配适应度："+s1+"  匹配个数为："+size(pair,1));
                    fg=figure();
                    pcshowMatchedFeatures(tp1,p2,pointCloud(ntp1.Location(d<yuzhi,:)),pointCloud(np2.Location(d<yuzhi,:)),'PlotOptions',{'bo','bo','b-'});

                    hold on;
                    pcshowMatchedFeatures(tp1,p2,pointCloud(ntp1.Location(d>yuzhi,:)),pointCloud(np2.Location(d>yuzhi,:)),'PlotOptions',{'r*','r*','r--'});
                    disp("");
                    set(fg,'color','w');
                    axis off;


                    % fg=figure();
                    % pcshowMatchedFeatures(p1,p2,pointCloud(np1.Location(d<yuzhi,:)),pointCloud(np2.Location(d<yuzhi,:)),'PlotOptions',{'bo','bo','b-'});
                    % hold on;
                    % pcshowMatchedFeatures(p1,p2,pointCloud(np1.Location(d>yuzhi,:)),pointCloud(np2.Location(d>yuzhi,:)),'PlotOptions',{'r*','r*','r--'});
                    % disp("");
                    % set(fg,'color','w');
                    % axis off;

                end
                %画热力图
                % if size(pair,1)>=3&&i==5&&j==6
                %     %根据Ransac求出T
                %     pts1=np1.Location;
                %     pts2=np2.Location;
                %     % set RANSAC Parameters
                %     % number of sampled points. minimum is 3 for rigid body transformation  每次采样的点数
                %     coeff.minPtNum = 3; 
                % 
                %     % number of iterations                                         % RANSAC 迭代次数（多少次随机取点）
                %     % coeff.iterNum = 2e3;
                %     coeff.iterNum=2000;
                % 
                %     % distance (in world units世界单位) below which matches are considered inliers  判断是内点的阈值
                %     coeff.thDist = yuzhi; 
                % 
                %     % percentage of matches that are inliers needed to call the transformation a success 称为转换成功所需的内部匹配的百分比，最后用来计算一个sucess rate
                %     coeff.thInlrRatio = 0.1; 
                %     %% Perform RANSAC with rigid transform T and distance function
                %     % REFINE: find the transformation again using all inliers, if successful 是否继续使用筛选后的内点再找一次转换
                %     coeff.REFINE = true;
                %     [TTT, ~] = ransac_methord(pts2,pts1,coeff,@estimateTransform,@calcDists);
                %     tp1=pctransform(p1,affine3d(TTT));
                %     % fg=figure();
                %     % p3=pcmerge(tp1,p2,0.01);
                %     % pcshow(p3);
                %     % set(fg,'color','w');
                %     % axis off;
                %     tform.Translation=TTT(4,1:3);
                %     gt_tform.Translation=T(4,1:3);
                %     tform.Rotation=TTT(1:3,1:3);
                %     gt_tform.Rotation=T(1:3,1:3);
                %     [Et,Er]=cal_error(tform,gt_tform);
                %     Ets(i,j)=Et;
                %     Ers(i,j)=Er;
                %     disp("第"+i+"对点云与第"+j+"对点云：");
                %     disp("Et："+Et+"  Er："+Er);
                % end
                %源点云和目标点云匹配效果
                if s1>0.6&&num>10
                    % disp();
    %                 disp("第"+i+"片点云与第"+j+"片点云匹配适应度："+s1+"  匹配个数为："+size(pair,1));
    %                 dopairpicture(p1,p2,pair,T,yuzhi)
    %                 fg=figure();
    %                 pcshowMatchedFeatures(p1,p2,pointCloud(np1.Location(d<yuzhi,:)),pointCloud(np2.Location(d<yuzhi,:)),'PlotOptions',{'bo','bo','b-'});
    %                 hold on;
    %                 pcshowMatchedFeatures(p1,p2,pointCloud(np1.Location(d>yuzhi,:)),pointCloud(np2.Location(d>yuzhi,:)),'PlotOptions',{'r*','r*','r--'});
    %                 set(fg,'color','w');
    %                 disp("");
                end
    
    
                %配准到一起效果
    %             if s1>0.6&&num>10
    %                 disp("第"+i+"片点云与第"+j+"片点云匹配适应度："+s1+"  匹配个数为："+size(pair,1));
    %                 fg=figure();
    %                 pcshowMatchedFeatures(tp1,p2,pointCloud(tp1.Location(d<yuzhi,:)),pointCloud(np2.Location(d<yuzhi,:)),'PlotOptions',{'bo','bo','b-'});
    %                 hold on;
    %                 pcshowMatchedFeatures(tp1,p2,pointCloud(tp1.Location(d>yuzhi,:)),pointCloud(np2.Location(d>yuzhi,:)),'PlotOptions',{'r*','r*','r--'});
    %                 disp("");
    %                 set(fg,'color','w');
    %             end
    
    %             if(s1>0.5)
    %                 pcshowMatchedFeatures(tpp,p2,tp1,np2);
    %             end
                
            end
        end
    end
    score=score/(pnum*xunhuannum);
    scores=scores/(xunhuannum);
    matchnum=matchnum/(xunhuannum);
    disp("多尺度平均适应度："+score);
end
function af=readAllRadiusF(r)
    for i=1:r
        af{i}=load("Data\feature\radiuF"+i+".mat").r;
    end
end
