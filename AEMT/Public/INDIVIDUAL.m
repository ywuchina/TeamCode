classdef INDIVIDUAL  
    % This class contains P populations.
    % Consists of four items: the number of pop, the index of pop,
    % chromosomes, function value.
    % INDIVIDUAL needs to be initialized by initPOP.
    properties
        pop;       % 种群个数
        idvs;      % 个体值 
        fits;      %适应度
        task;      % Function value
        pc;  %杂交概率
        pm;  %变异概率
        elite_num; %精英个体数
        prm; %参数
    end    
    methods        
        function object = initINDIVIDUAL(object,prm,pop,task,pc,pm,elite_num,W)
            Maxnum=prm.Maxnum;yuzhi=prm.yuzhi;dim=prm.dim;LEN=prm.LEN;T=prm.T;p=prm.p;F=prm.F;data=prm.data;GrtT=prm.GrtT;GrtR=prm.GrtR;
    
            object.pop=pop;
            object.prm=prm;
            object.task=task;
            object.pc=pc;
            object.pm=pm;
            object.pop=pop;
            object.elite_num = elite_num;
            idvs=rand(pop,dim);
            idvs(:,task.space==0)=0;
            idvs(idvs>(1-W))=1;%权重越大被选择的概率越大
            idvs(idvs<=(1-W))=0;
            object.idvs=idvs;
            object.fits=eva4(idvs,prm);
        end
        function o=mutation(Individual,o1)
            pm=Individual.pm;
            task=Individual.task;
            space=task.space;
            o=zeros(size(o1));
            for i=1:size(o1,1)
                o(i,:)=o1(1,:);
                for j=1:size(o1,2)
                    if space(j)==0
                        continue;
                    end
                    r=rand(1,1);
                    if r<pm
                        o(i,j)=1-o1(i,j);
                    end
                end
            end
            end
        function o=cross(Individual,pairs)
        %%双点交叉杂交
            idvs=Individual.idvs;
            p=Individual.prm.p;
            o=zeros(size(idvs));
            dim=size(idvs,2);
            num=0;
            for i=1:size(pairs,1)
                r=rand(1,1);
                if r>p
                    continue;
                end
                p1=idvs(pairs(i,1),:);
                p2=idvs(pairs(i,2),:);
            %     rr=ceil(rand(1,2)*dim);
            %     r1=min(rr);%交换起始位置
            %     r2=max(rr);%交换终止位置
                len=20;%交换长度
                r1=ceil(rand(1,2)*(dim-len));
                r2=r1+len;
                c1=p1;c1(r1:r2)=p2(r1:r2);
                c2=p2;c2(r1:r2)=p1(r1:r2);
                num=num+1;
                o(num,:)=c1;
                num=num+1;
                o(num,:)=c2;
            end
            o=o(1:num,:);
        end
        function pairs=generatePairs(Individual)
            %根据轮盘赌生成pop/2对父个体
            pop=Individual.pop;
            fits=Individual.fits;
            s=0;
            all=sum(fits);
            area=zeros(pop,1);
            for i=1:pop
                s=s+fits(i)/all;
                area(i)=s;
            end
            
            pairs=zeros(pop/2,2);
            for i=1:floor(pop/2)
                r1=rand(1,1);
                r2=rand(1,1);
                for j=1:pop
                    if r1<=area(j)
                        pairs(i,1)=j;
                        break;
                    end
                end
                if pairs(i,1)==0
                    pairs(i,1)=ceil(rand(1)*pop);
                end
                for j=1:pop
                    if r2<=area(j)
                        pairs(i,2)=j;
                        break;
                    end
                end
                if pairs(i,2)==0
                    pairs(i,2)=ceil(rand(1)*pop);
                end
            end
        end
        function [idvs,nfits]=select(Individual,all,fits)
            pop=Individual.pop;
            elite_num=Individual.elite_num;
            nfits=zeros(pop,1);
            [~,idx]=sort(-fits);
            idvs=zeros(pop,size(all,2));
            idvs(1:elite_num,:)=all(idx(1:elite_num),:);%选择精英个体
            nfits(1:elite_num)=fits(idx(1:elite_num));
            s=0;
            ALL=sum(fits);
            area=zeros(size(all,1),1);
%             area=ones(size(all,1),1)/size(all,1);
%             for i=2:size(all,1)
%                 area(i)=area(i)+area(i-1);
%             end
            for i=1:size(all,1)
                s=s+fits(i)/ALL;
                area(i)=s;
            end
            
            for i=elite_num+1:pop%轮盘选择
                r=rand(1,1);
                for j=1:size(all,1)
                    if r<=area(j)
                        idvs(i,:)=all(j,:);
                        nfits(i)=fits(j);
                        break;
                    end
                end
            end
        end
        function object = generateNewIndividualFromOld(object,ntask,W)
            %ntasks是新任务 如果某维特征空间新老任务都有，则直接继承下来如果新的有原来没有，则根据权重生成
            idv=object.idvs&ntask.space;
            for i=1:size(object.idvs,1)%对每个个体操作
                r=rand(1,size(ntask.space,2));
                X=(~object.task.space)&ntask.space;%新的有但是老的没有的位置
                r(r<W)=0;
                r(r>=W)=1;
                r=1-r;
                idv(i,X==1)=r(1,X==1);
                object.idvs(i,ntask.space&r<W)=1;
            end
            object.fits=eva4(object.idvs,object.prm);
            object.task=ntask;
        end
    end
end