clc;
clear;
addpath(genpath(pwd));

prm=PARAMETER();
prm=initPARAMETER(prm);%初始化参数

d=DB();%初始化数据库
d=initDB(d,prm);

prm.st=generateSTD(prm);
init(prm);%对数据做处理

%需要查看d中的pair数
[scoress,matchnums,fsnums,times]=testMain(prm,d);
scoress=scoress';
matchnums=matchnums';



function [scoress,matchnums,fsnums,times]=testMain(prm,d)
    Maxnum=prm.Maxnum;yuzhi=prm.yuzhi;dim=prm.dim;LEN=prm.LEN;T=prm.T;p=prm.pp;F=prm.FF;data=prm.data;GrtT=prm.GrtT;GrtR=prm.GrtR;
    
    %找最优解
    p=0;
    times=[];
    fsnums=[];

    % ReliefF
    disp("ReliefF start");
    p=p+1;
    [best(p,:),times(1,p)]=calcRelieff(d,prm);
    fsnums(1,p)=sum(best(p,:));

    %皮尔逊相关系数
    disp("Pearson start");
    p=p+1;
    [best(p,:),times(1,p)]=calcperson(d,prm);
    fsnums(1,p)=sum(best(p,:));

    %互信息
    disp("Mutual Information start");
    p=p+1;
    [best(p,:),times(1,p)]=calcMI(d,prm);
    fsnums(1,p)=sum(best(p,:));

    %CUSPSO
    disp("CUPSO start");
    p=p+1;
    [best(p,:),times(1,p)]=calcCUS(d,prm);
    fsnums(1,p)=sum(best(p,:));

    %MFCSO
    disp("MFCSO start");
    p=p+1;
    [best(p,:),times(1,p)]=calcMFCSO(d,prm);
    fsnums(1,p)=sum(best(p,:));

    disp("AEMT start");
    p=p+1;
    [best(p,:),times(1,p)]=method(prm,d);   %我们的方法找到最优解
    fsnums(1,p)=sum(best(p,:));
    
    % best=load('allbest.mat').best;
    % best=best(10,:);

    p=0;
    scoress=[];
    matchnums=[];
    nums=[];
    xunhuannum=1;
    scores=0;
    matchnum=0;

    disp("ReliefF PCC 互信息 CUSPSO MFCSO 本方法：");
    for i=1:size(best,1)
        [matchnum,scores]=f1(prm,xunhuannum,best(i,:));
        % disp(sum(scores>0.5)+" "+sum(matchnum/size(matchnum,2)));
        p=p+1;
        scoress(p,:)=scores;
        matchnums(p,:)=matchnum;
    end
end
function [matchnum,scorearr]=f1(prm,xunhuannum,X)
    
    s1=0;
    scorearr=[];
    for j=1:xunhuannum
        [mnum,ts,tarr]=test(prm,X);
        if isempty(scorearr)
            scorearr=tarr;
            matchnum=mnum;
        else
            scorearr=scorearr+tarr;
            matchnum=matchnum+mnum;
        end
        
        s1=s1+ts; %测试效果

    end
    s1=s1/xunhuannum;
    score=s1;
    scorearr=scorearr/xunhuannum;
    disp("平均得分："+score);

%     s3=0;
%     for j=1:xunhuannum
%         s3=s3+test(prm,best); %测试效果
%     end
%     s3=s3/xunhuannum;
%     disp("选择的描述子测试集平均适应度："+s3);
    
end