function [LBEST,gbest,AD,W,Sample,M,pop,flag]=ppso1(pop,LBEST,popsize,k,X0,gbest,mm,u,W,Sample,M,AD_0,flag)
%--------样本子集下评价个体
% PopEff0=cell(M,1);
PopEff=cell(M,1);
%  %----------------------------------
for I=1:M
    PopEff{I}=evaluation(pop,popsize,mm,Sample{I},u,k);%每个元胞的第mm+1行是第M个样本子集上的分类精度，并且所有元胞mm+2行结果一样是粒子包含特征数目
end
% ------利用权重集成最后的适应值

popeff=zeros(popsize,mm+3+M);%mm+3放粒子的不确定度，mm+1放加权后近似适应值，mm+3+1：mm+3+M存放M个子适应值。
popeff(:,1:mm)=pop;
%-------求粒子的加权后的近似适应值和不确定度
for i=1:popsize
    ary=0;
    av=0;%近似适应值均值
    Uncertainty=0;%不确定度
    sub_eff=[];%用来保存每个个体在每个样本子集上的适应值
    for j=1:M
        a_f=PopEff{j}(i,mm+1);
        ary=ary+W(j)*a_f;
        av=av+a_f;
        sub_eff=[sub_eff,a_f];%最终形式是一行M列。
    end
    av=av/M;
    for j=1:M
        a_f=PopEff{j}(i,mm+1);
        Uncertainty=Uncertainty+(a_f-av)^2;
    end
    Uncertainty=(Uncertainty/(M-1))^0.5;
    
    popeff(i,mm+1)=ary;%加权后适应值
    popeff(i,mm+2)= PopEff{1}(i,mm+2);
    popeff(i,mm+3)=Uncertainty;
    popeff(i,mm+4:end)=sub_eff;%从mm+4列存放第一个样本子集上的适应值
    
end
%------------个体极值点的更新(基于加权后适应值的基础上)

for i=1:popsize
    if popeff(i,mm+1)>LBEST(i,mm+1) 
        LBEST(i,1:mm+2+M)=[popeff(i,1:mm+2),popeff(i,mm+4:end)];
    elseif popeff(i,mm+1)==LBEST(i,mm+1)&&popeff(1,mm+2)<LBEST(i,mm+2)
        LBEST(i,1:mm+2+M)=[popeff(i,1:mm+2),popeff(i,mm+4:end)];
    end
end

%------------  全局极值点的更新
[~,index]=max(LBEST(:,mm+1));
best=LBEST(index,1:mm+2);%在近似适应值下最大，best的第mm+1位放近似适应值
%--------计算真实适应值
[aa,~]=kmean_res(best(1:mm),X0,u,k,mm);
zhen_best=best;%在近似适应值下最大，zhen_best的第mm+1位放加权后近似适应值，zhen_best的第mm+2位放特征规模
zhen_best(mm+1)=aa*100;%zhen_best的第mm+1位放的是真实适应值
if zhen_best(1,mm+1)>gbest(1,mm+1)
    gbest(1,1:mm+2)=zhen_best(1,1:mm+2);%%%%%-----在整个进化过程中， gbest的小尾巴都放真实适应值
    %%-----加权适应值，为了后面判断模型是否更新
    gbest(1,mm+3)=best(1,mm+1);
    
elseif zhen_best(1,mm+1)==gbest(1,mm+1)&&zhen_best(1,mm+2)<gbest(1,mm+2)
    gbest(1,1:mm+2)=zhen_best(1,1:mm+2);
    %%-----加权适应值，为了后面判断模型是否更新
    gbest(1,mm+3)=best(1,mm+1);
end


%------------------------- 本次迭代通用样本点，
%-------权重更新储备集AD的填充：1近似最优解，2不确定度最大解
%---------权重更新储备集,1:mm列放粒子，mm+1放加权后适应值，mm+2放特征数目,mm+3放真实适应值
DD=[];
%--近似适应值最大的粒子---BBest:mm+1是近似适应值，mm+2是特征规模，mm+3是真实适应值

BBest=[best,aa*100];%近似适应值最大的粒子---BBest:mm+1是近似适应值，mm+2是特征规模，mm+3是真实适应值

%--不确定度最大的粒子
[~,index]=max(popeff(:,mm+3));
UBest=popeff(index,1:mm+2);%第mm+1是加权中近似适应值
%--------计算真实适应值
[b,~]=kmean_res(UBest(1:mm),X0,u,k,mm);
UBest=[UBest,b*100];%mm+3是真实适应值

%-----本次迭代将BBest和UBest放入权重更新储备集
DD=[BBest;UBest];
%---------------------------------
    %-------确定集成代理权重更新时机
     if flag==5 || flag>5
            %------环境变化，需要重新更新初始种群pop，更新权重W， 更新当前LBEST,gbest,更新AD
            M_0=1;
            M=M-round(M_0);
            kk=size(X0,1);
            Sample=Sample_grouping(kk,X0,M);
            %----------------------------------
            %-------更新当前种群
            %-------一部分需要重置
            %---------找到上一时刻权重下，LBEST加权后适应值最小的5%的个体进行随机初始化、
            popsize1=ceil(0.05*popsize);
            pop1=zeros(popsize1,mm);
            for i=1:(popsize1)
                for j=1:mm
                    pop1(i,j)=randperm(length(u{j})+1,1)-1;
                    %-----和当前最优解交叉变异
                    pop1(i,j)=round(0.5*(pop1(i,j)+gbest(1,j)));
                end
            end
            [~,index]=sort(LBEST(:,mm+1));%升序排列
            set=index(1:popsize1);
            LBEST(set,:)=[];
            pop=[pop1;LBEST(:,1:mm)];
            %---------------------------------------
            %--------新的种群和样本子集下重新近似评价当前环境的初始种群
            sub_popeff=cell(M,1); %用元胞来保存在M个样本子集上的种群分类精度
            pop_0=pop(:,1:mm);%初始种群
            
            %-----------------------
            pop=pop_0;
            % parpool 'local' 'MATLAB Parallel Cloud';
            for i=1:M
                sub_popeff{i}=evaluation(pop_0,popsize,mm,Sample{i},u,k);%当前样本子集划分下，种群在每个样本子集下的适应值
            end
            % parpool closed;
            %----------------------------------

            %-------更新更新权重
            %-------(这种方式需要重新计算当前新种群pop的真实适应值，耗时)重新更新储备集合=当前新种群pop+DD(当代最优+不确定性最大)+DB(每个样本子集下的最优粒子)
            %-------重新更新储备集合=第一代初始种群pop(真实适应值已经被计算，省时)+DD(当代最优+不确定性最大)+DB(每个样本子集下的最优粒子)
            AD=[AD_0(1:popsize,:);DD];
            %-----还需要补充最新样本集合划分下每个样本子集下的最优粒子
            DB=[];

            for s=1:M
                pp=sub_popeff{s};
                [~,index1]=max(pp(:,mm+1));
                Sub_best=pp(index1,1:mm+2);%在近似适应值下最大，Sub_best的第mm+1位放是上一个样本划分下近似适应值，还要被新的样本划分修改
                [sb,~]=kmean_res(Sub_best(1:mm),X0,u,k,mm);%要被真实评价
                Sub_best=[Sub_best,sb*100];%mm+3是真实适应值
                DB=[DB;Sub_best];
            end
            AD=[AD;DB];
            %---------------------------------------

            %----样本重新划分后更新权重
            popeff_0=[AD(:,1:mm),AD(:,mm+3),AD(:,mm+2)];
            ppsize=size(popeff_0,1);
            W=sub_model_1(popeff_0,Sample,k,M,mm,u,ppsize);%----样本划分时，同时调整W。
            %------利用权重集成最后的适应值
            w_eff=zeros(popsize,mm+2+M);%mm+1放加权后近似适应值
            w_eff(:,1:mm)=pop_0;
            %-------求粒子的加权后的近似适应值,以及在每个样本子集上的适应值
            for i=1:popsize
                ary=0;
                sub_eff=[];%用来保存每个个体在每个样本子集上的适应值
                for j=1:M
                    a_f=sub_popeff{j}(i,mm+1);
                    sub_eff=[sub_eff,a_f];%最终形式是一行M列。
                    ary=ary+W(j)*a_f;
                end

                w_eff(i,mm+1)=ary;
                w_eff(i,mm+2)= sub_popeff{1}(i,mm+2);
                w_eff(i,mm+3:end)=sub_eff;%从mm+3列存放第一个样本子集上的适应值
            end

            %---------初始极值点的更新
            LBEST=w_eff;%里面不存在真实适应值，mm+1加权适应值，mm+2特征子集规模，从mm+3列存放第一个样本子集上的适应值
            %--------新的种群和样本子集下重新设置gbest
            %------------  全局极值点的更新
            [~,index]=max(LBEST(:,mm+1));
            best=LBEST(index,1:mm+2);%在近似适应值下最大，best的第mm+1位放近似适应值
            %--------计算真实适应值
            [aa,~]=kmean_res(best(1:mm),X0,u,k,mm);
            zhen_best=best;%在近似适应值下最大，zhen_best的第mm+1位放加权后近似适应值，zhen_best的第mm+2位放特征规模
            zhen_best(mm+1)=aa*100;%zhen_best的第mm+1位放的是真实适应值
            if zhen_best(1,mm+1)>gbest(1,mm+1)
                gbest(1,1:mm+2)=zhen_best(1,1:mm+2);%%%%%-----在整个进化过程中， gbest的小尾巴都放真实适应值
                %%-----加权适应值，为了后面判断模型是否更新
                gbest(1,mm+3)=best(1,mm+1);

            elseif zhen_best(1,mm+1)==gbest(1,mm+1)&&zhen_best(1,mm+2)<gbest(1,mm+2)
                gbest(1,1:mm+2)=zhen_best(1,1:mm+2);
                %%-----加权适应值，为了后面判断模型是否更新
                gbest(1,mm+3)=best(1,mm+1);
            end
            %---------------------------------------------------
            %-------环境变化后，则重置flag
            flag=0;
            
            
            
            %               flag=0;G_feal=gbest(mm+1);
            %               G_feal=gbest(mm+1);
     else
            %------如果上一步重新划分样本了，那么权重就已经被重新更新了,不需要重复执行权重更新机制
            %------没有重新划分样本，判断是否更新权重
            %------最后更新权重更新时刻的全局极值点和个体极值点
            AD=[AD_0;DD];%----储备集填充
            %------判断代理模型误差
            E=abs(gbest(mm+3)-gbest(mm+1))/(gbest(mm+1));
            if E>0.01  
                %-------更新权重(注意：此刻的AD_0，已经在子函数ppso1.m中被更新过，不重复补充)
                popeff_0=[AD(:,1:mm),AD(:,mm+3),AD(:,mm+2)];
                ppsize=size(popeff_0,1);
                W=sub_model_1(popeff_0,Sample,k,M,mm,u,ppsize);
                %---------------------------------------------

                %--------新的权重下更新加权适应值，由于样本子集没有重新划分，只需要叠加新的权重计算加权后适应值,
                for i=1:popsize
                    ary=0;
             
                    for j=1:M
                        a_f=LBEST(i,mm+2+j);
                        ary=ary+W(j)*a_f;
                    end
                    
                    LBEST(i,mm+1)=ary;%---------第mm+1列是加权适应值。
                   
                end
                %-------------------------------------------------------

            end
            %-------------------------------------
     end


       