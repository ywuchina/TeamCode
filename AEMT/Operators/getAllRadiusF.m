function getAllRadiusF()
%读取所有的特征
    rnum=20;%半径数量
    Maxnum=20;%点云的数量
    af=cell(0);
    for i=1:rnum
        af{i}=getF(i,Maxnum);
    end
    save("Data\feature\allRadiusF.mat","af");
    for i=1:20
        r=af{i};
        save("Data\feature\radiuF"+i+".mat","r");
    end
end
function F=getF(r,Maxnum)
    pallF=cell(0);
    FF=cell(0);
    AllF=[];
    for i=1:Maxnum
%         str1="feature\p"+i+"_r20_";
        str1="Data\feature\p"+i+"_r"+r+"_";
        [FF{i},LEN]=generateF(str1,i);
        AllF=[AllF;FF{i}];
    end
%     save("Data\feature\LEN.mat",'LEN');
    %归一化
    x=AllF;
    Max=max(x);
    Min=min(x);
    for i=1:Maxnum
        f=FF{i};
        for j=1:length(Max)
            if (Max(j)-Min(j))==0
                f(:,j)=zeros(length(f(:,j)),1);
            else
                f(:,j)=(f(:,j)-Min(j))/(Max(j)-Min(j));
            end
        end
        pallF{i}=f;
    end
    F=pallF;
end
function [F,LEN]=generateF(str1,i)
    str2=".txt";
%     fpfhF=processFPFHData(importdata(str1+'fpfh'+str2));save("feature\fpfhF_p"+i+".mat",'fpfhF');
%     %pfhF=processPFHData(importdata(str1+'pfh'+str2));save("feature\pfhF_1"+i+".mat",'pfhF');
%     %rsdF=processRSDData(importdata(str1+'rsd'+str2));save("feature\rsdF_"+i+".mat",'rsdF');
%     %uscF=processUSCData(importdata(str1+'usc'+str2));
%     shotF=processShotData(importdata(str1+'shot'+str2));save("feature\shotF_p"+i+".mat",'shotF');
%     siF=processSpinImageData(importdata(str1+'SpinImage'+str2));save("feature\siF_p"+i+".mat",'siF');
%     ropsF=processRopsData(importdata(str1+'rops'+str2));save("feature\ropsF_p"+i+".mat",'ropsF');

    fpfhF=processFPFHData(importdata(str1+'fpfh'+str2));
    %pfhF=processPFHData(importdata(str1+'pfh'+str2));save("Data\feature\p"+i+"_pfhF.mat",'pfhF');
    %rsdF=processRSDData(importdata(str1+'rsd'+str2));save("Data\feature\p"+i+"_rsdF.mat",'rsdF');
    %uscF=processUSCData(importdata(str1+'usc'+str2));
    shotF=processShotData(importdata(str1+'shot'+str2));
    siF=processSpinImageData(importdata(str1+'SpinImage'+str2));
    ropsF=processRopsData(importdata(str1+'rops'+str2));

    F=[fpfhF,shotF,siF,ropsF];
    LEN=[length(fpfhF(1,:)),length(shotF(1,:)),length(siF(1,:)),length(ropsF(1,:))];

    
end
function feature=processRopsData(data)
    data=split(data,',');
    t=split(data(:,135),')');
    data(:,135)=t(:,1);
    t=split(data(:,1),'(');
    data(:,1)=t(:,2);
    [r,l]=size(data);
    feature=ones(r,l);
    for i=1:length(data)
        for j=1:length(data(1,:))
            t=data(i,j);
            t=cell2mat(t);
            t=str2double(t);
            feature(i,j)=t;
        end
        
    end
end

function feature=processSpinImageData(data)
    data=split(data,',');
    for i=1:length(data(:,1))
        if data(i,1)=="(-nan(ind)"
            data(i,1)={'(0'};
            data(i,length(data(i,:)))={'0)'};
            for j=2:length(data(i,:))-1
                data(i,j)={'0'};
            end            
            continue;
        end
    end
    t=split(data(:,153),')');
    data(:,153)=t(:,1);
    t=split(data(:,1),'(');
    data(:,1)=t(:,2);
    [r,l]=size(data);
    feature=ones(r,l);
    for i=1:length(data)
        for j=1:length(data(1,:))
            t=data(i,j);
            t=cell2mat(t);
            t=str2double(t);
            feature(i,j)=t;
        end
        
    end
end
function feature=processShotData(data)
    data=split(data,',');
    data=data(:,10:360);
    t=split(data(:,351),')');
    data(:,351)=t(:,1);
    [r,l]=size(data);
    feature=ones(r,l);
    for i=1:r
        for j=1:l
            t=data(i,j);
            t=cell2mat(t);
            t=str2double(t);
            feature(i,j)=t;
        end
        
    end
end
function feature=processRSDData(data)
    data=split(data,',');
    
    t=split(data(:,2),')');
    data(:,2)=t(:,1);
    t=split(data(:,1),'(');
    data(:,1)=t(:,2);
    [r,l]=size(data);
    feature=ones(r,l);
    for i=1:length(data)
        for j=1:length(data(1,:))
            t=data(i,j);
            t=cell2mat(t);
            t=str2double(t);
            feature(i,j)=t;
        end
        
    end
end
function feature=processFPFHData(data)
    data=split(data,',');
    t=split(data(:,33),')');
    data(:,33)=t(:,1);
    t=split(data(:,1),'(');
    data(:,1)=t(:,2);
    [r,l]=size(data);
    feature=ones(r,l);
    for i=1:length(data)
        for j=1:length(data(1,:))
            t=data(i,j);
            t=cell2mat(t);
            t=str2double(t);
            feature(i,j)=t;
        end
        
    end
end
function feature=processPFHData(data)
    data=split(data,',');
    t=split(data(:,125),')');
    data(:,125)=t(:,1);
    t=split(data(:,1),'(');
    data(:,1)=t(:,2);
    [r,l]=size(data);
    feature=ones(r,l);
    for i=1:length(data)
        for j=1:length(data(1,:))
            t=data(i,j);
            t=cell2mat(t);
            t=str2double(t);
            feature(i,j)=t;
        end
        
    end
end
function feature=processUSCData(data)
    data=split(data,')');
    t=split(data(:,2),'(');
    data=split(t(:,2),',');
    [r,l]=size(data);
    feature=ones(r,l);
    for i=1:size(data,1)
        for j=1:length(data(1,:))
            t=data(i,j);
            t=cell2mat(t);
            t=str2double(t);
            feature(i,j)=t;
        end
        
    end
end
