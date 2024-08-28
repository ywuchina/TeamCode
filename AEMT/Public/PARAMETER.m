classdef PARAMETER    
    %This class contains all parameters and needs to be initialized by
    %initPARAMETER.
    properties
        Maxnum;  %点云数量
        yuzhi;   %阈值
        dim;     %维度
        LEN;     %每个描述子的维度
        T;       %旋转矩阵
        p;       %下采样两次的点云
        F;       %特征
        data;    %点云数据
        GrtT;    %平移
        GrtR;    %旋转
        pp;      %下采样点云
        FF;      %下采样特征
        st;      %特征的标准差
        overlap; %计算点云间重叠率
        originoP;%原始点云
        dataname;%数据集名字
        name;%模型名字
    end    
    methods        
        function object = initPARAMETER(object)
            dataname="WHU";
            name="Campus";
            object.name=name;
            object.dataname=dataname;
            load("Data\PC\"+dataname+'-'+name+".mat");
            object.GrtR=GrtR;
            object.GrtT=GrtT;
            object.data=data;
            % object.originoP=originp;

            %data数量大于24才需要处理
            object.GrtR=object.GrtR(1,1:6);
            object.GrtT=object.GrtT(1,1:6);
            object.data=object.data(1,1:6);
            % 
            object.p=load("Data\PC\ddp.mat").dp;
            object.F=load("Data\PC\ddp.mat").df;
            object.pp=load("Data\PC\p.mat").p;
            object.FF=load("Data\feature\AllFeature").pallF;

            object.Maxnum=size(object.pp,2);
            object.Maxnum=6;
            object.yuzhi=cal_yuzhi(object.pp(1).Location');
            object.LEN=load("feature\LEN.mat").LEN;
            object.dim=sum(object.LEN);

            for i=1:object.Maxnum
                t=[object.GrtR{i},object.GrtT{i}];
                t=[t;0 0 0 1];
                t=t';
                TT{i}=affine3d(t);
            end
            object.T=TT;

            ol=[];
            for i=1:object.Maxnum
                for j=1:object.Maxnum
                    if i==j
                        object.overlap(i,j)=1;
                        continue;
                    end
                    t=[object.GrtR{i},object.GrtT{i}];
                    t=[t;0 0 0 1];
                    t=t';
                    T1=t;
                    t=[object.GrtR{j},object.GrtT{j}];
                    t=[t;0 0 0 1];
                    t=t';
                    T2=t;
                    TTT=round(T1/T2,10); 
                    p1=object.pp(i);
                    p2=object.pp(j);
                    tp1=pctransform(p1,affine3d(TTT));
                    ol(i,j)=calcOverlap(p1,p2,TTT,object.yuzhi);
                    % pcshowpair(tp1,p2);
                end
            end
            object.overlap=ol;
        end  
    end
end