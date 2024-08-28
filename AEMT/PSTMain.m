clc;
clear;
addpath(genpath(pwd));

prm=PARAMETER();
prm=initPARAMETER(prm);%初始化参数
scoress=[];
matchnums=[];
i=1;
for proportion=0.4:0.1:1
    j=1;
    for G=5:5:30
        d=DB();%初始化数据库
        d=initDB(d,prm);
        prm.st=generateSTD(prm);
        % init(prm);%对数据做处理
        %需要查看d中的pair数
        [score,matchnum]=testMain2(prm,d,G,proportion);
        scoress(i,j)=score;
        matchnums(i,j)=matchnum;
        j=j+1;
    end
    i=i+1;
end



function [score,matchnum]=testMain2(prm,d,G,proportion)
    Maxnum=prm.Maxnum;yuzhi=prm.yuzhi;dim=prm.dim;LEN=prm.LEN;T=prm.T;p=prm.pp;F=prm.FF;data=prm.data;GrtT=prm.GrtT;GrtR=prm.GrtR;
    
    %找最优解
    best=PSTmethod(prm,d,proportion,G);   %我们的方法找到最优解

    p=0;
    score=0;
    matchnum=0;
    xunhuannum=30;

    %普通选择的特征效果测试
    disp("主特征所占比例："+proportion+"，进化代数为："+G+"本方法精度：");
    [matchnum,score]=f1(prm,xunhuannum,best);
    disp(score);
    
end


function [scoress,matchnums,fsnums,times]=testMain(prm,d)
    Maxnum=prm.Maxnum;yuzhi=prm.yuzhi;dim=prm.dim;LEN=prm.LEN;T=prm.T;p=prm.pp;F=prm.FF;data=prm.data;GrtT=prm.GrtT;GrtR=prm.GrtR;
    
    %找最优解
    p=0;
    times=[];
    fsnums=[];



    %ReliefF
    % p=p+1;
    % [best(p,:),times(1,p)]=calcRelieff(d,prm);
    % fsnums(1,p)=sum(best(p,:));
    % 
    % %皮尔逊相关系数
    % p=p+1;
    % [best(p,:),times(1,p)]=calcperson(d,prm);
    % fsnums(1,p)=sum(best(p,:));
    % 
    % %互信息
    % p=p+1;
    % [best(p,:),times(1,p)]=calcMI(d,prm);
    % fsnums(1,p)=sum(best(p,:));
    % 
    % %互信息和皮尔逊一起用
    % % p=p+1;
    % % [best(p,:),times(1,p)]=calcPCCAndMI(d,prm);
    % % fsnums(1,p)=sum(best(p,:));
    % 
    % %CUSPSO
    % p=p+1;
    % [best(p,:),times(1,p)]=calcCUS(d,prm);
    % fsnums(1,p)=sum(best(p,:));
    % 
    % %MFCSO
    % p=p+1;
    % [best(p,:),times(1,p)]=calcMFCSO(d,prm);
    % fsnums(1,p)=sum(best(p,:));

    p=p+1;
    [best(p,:),times(1,p)]=nomultitask(prm,d);   %没有多任务
    fsnums(1,p)=sum(best(p,:));

    p=p+1;
    [best(p,:),times(1,p)]=method(prm,d);   %我们的方法找到最优解
    fsnums(1,p)=sum(best(p,:));


    p=p+1;
    prm.p=prm.pp;
    prm.F=prm.FF;
    [best(p,:),times(1,p)]=method(prm,d);   %没有代理模型
    fsnums(1,p)=sum(best(p,:));

    

    %加载最好的解
    % best=load("best.mat").best;
    % % [best,~]=method(prm,d);
    % p=size(best,2);
    %————————————————————

    p=0;
    scoress=[];
    matchnums=[];
    nums=[];
    xunhuannum=1;
    scores=0;
    matchnum=0;
    
    %描述子本身效果测试
    disp("四个描述子效果：");
    l=1;r=33;
    for i=1:4
        X=zeros(1,672);
        X(1,l:r)=1;
        [matchnum,scores]=f1(prm,xunhuannum,X);%平均得分以及得分数组
        disp(sum(scores>0.5)+" "+sum(matchnum/size(matchnum,2)));
        if i~=4
            l=r+1;
            r=l+LEN(i+1)-1;
        end
        p=p+1;
        scoress(p,:)=scores;
        matchnums(p,:)=matchnum;
    end

    %普通选择的特征效果测试
    disp("ReliefF PCC 互信息 CUSPSO 本方法：");
    for i=1:size(best,1)
        [matchnum,scores]=f1(prm,xunhuannum,best(i,:));
        disp(sum(scores>0.5)+" "+sum(matchnum/size(matchnum,2)));
        p=p+1;
        scoress(p,:)=scores;
        matchnums(p,:)=matchnum;
    end

    %多尺度效果测试
    % disp("多尺度效果：");
    % for i=1:size(best,1)
    %     [matchnum,scores]=testMultiscale(prm,xunhuannum,best(i,:));
    %     disp(sum(scores>0.5)+" "+sum(matchnum/size(matchnum,2)));
    % 
    %     p=p+1;
    %     scoress(p,:)=scores;
    %     matchnums(p,:)=matchnum;
    % end
    
    p=0;
    Maxnum=prm.Maxnum;
    row=size(scoress,1);
    for i=1:Maxnum
        for j=i+1:Maxnum
            p=p+1;
            scoress(row+1,p)=i;
            scoress(row+2,p)=j;
            matchnums(row+1,p)=i;
            matchnums(row+2,p)=j;
        end
    end
    
end

function [matchnum,score]=f1(prm,xunhuannum,X)
    matchnum=0;
    score=0;
    for j=1:xunhuannum
        [mnum,s]=test2(prm,X);
        matchnum=matchnum+mnum;
        score=score+s;
    end
    matchnum=matchnum/xunhuannum;
    score=score/xunhuannum;
    disp("平均得分："+score);
    
end
function [matchnum,scoress]=test2(prm,best)
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
    for i=4:4
        for j=5:5
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
            num=num+1;
            score=score+s1;
        end
    end
end
