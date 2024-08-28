function getCorrAndRegFig(prm, X, i, j)
%测试多尺度效果
%   X是特征子集
    Maxnum=prm.Maxnum;yuzhi=prm.yuzhi;dim=prm.dim;LEN=prm.LEN;T=prm.T;p=prm.pp;F=prm.FF;data=prm.data;GrtT=prm.GrtT;GrtR=prm.GrtR;
    r=7;

    t=[GrtR{i},GrtT{i}];
    t=[t;0 0 0 1];
    t=t';
    T1=t;
    t=[GrtR{j},GrtT{j}];
    t=[t;0 0 0 1];
    t=t';
    T2=t;
    T=round(T1/T2,10);
    af=readAllRadiusF(r);
    for q=1:r
        f1{q}=af{q}{i};
        f2{q}=af{q}{j};
    end
    
    p1 = p(i); p2 = p(j);
    op1=pointCloud(prm.data{i}');op2=pointCloud(prm.data{j}');
    p1.Color = 'blue';
    p2.Color = 'cyan';

    op1.Color = 'blue';
    op2.Color = 'cyan';
    
    %找到匹配对
    pair=findpairs(p1,p2,f1,f2,X,yuzhi); 

    np1=pointCloud(p1.Location(pair(:,1),:));
    np2=pointCloud(p2.Location(pair(:,2),:));
    tp1=pctransform(np1,affine3d(T));
    
    d=[];
    for q=1:length(tp1.Location(:,1))
        d(q)=rms(tp1.Location(q,:)-np2.Location(q,:));
    end
    
   

    % p1=prm.originoP(i);
    % p2=prm.originoP(j);
    tp1=pctransform(p1,affine3d(T));
    ntp1=pctransform(np1,affine3d(T));

    otp1 = pctransform(op1,affine3d(T));
    p3=pcmerge(otp1,op2,0.01);
    

    fg1=figure();
    pcshow(op1);
    set(fg1,'color','w');
    axis off;


    fg2=figure();
    pcshow(op2);
    set(fg2,'color','w');
    axis off;

    fg3=figure();
    pcshowMatchedFeatures(op1,op2,pointCloud(np1.Location(d<yuzhi,:)),pointCloud(np2.Location(d<yuzhi,:)),'PlotOptions',{'go','go','g-'});
    hold on;
    pcshowMatchedFeatures(op1,op2,pointCloud(np1.Location(d>yuzhi,:)),pointCloud(np2.Location(d>yuzhi,:)),'PlotOptions',{'r*','r*','r--'});
    set(fg3,'color','w');
    axis off;

    fg4=figure();
    pcshowMatchedFeatures(otp1,op2,pointCloud(ntp1.Location(d<yuzhi,:)),pointCloud(np2.Location(d<yuzhi,:)),'PlotOptions',{'go','go','g-'});
    hold on;
    pcshowMatchedFeatures(otp1,op2,pointCloud(ntp1.Location(d>yuzhi,:)),pointCloud(np2.Location(d>yuzhi,:)),'PlotOptions',{'r*','r*','r--'});
    disp("");
    set(fg4,'color','w');
    axis off;

    fg5=figure();
    pcshow(p3);
    set(fg5,'color','w');
    axis off;
    % disp("第"+i+"片点云与第"+j+"片点云匹配适应度："+s1+"  匹配个数为："+size(pair,1));
    
    saveas(fg1,"picture\CorrAndReg\"+prm.name + '\1.png');
    saveas(fg2,"picture\CorrAndReg\"+prm.name + '\2.png');
    saveas(fg3,"picture\CorrAndReg\"+prm.name + '\3.png');
    saveas(fg4,"picture\CorrAndReg\"+prm.name + '\4.png');
    saveas(fg5,"picture\CorrAndReg\"+prm.name + '\5.png');
    
end
function af=readAllRadiusF(r)
    for i=1:r
        af{i}=load("Data\feature\radiuF"+i+".mat").r;
    end
end
