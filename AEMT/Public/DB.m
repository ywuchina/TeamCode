classdef DB    
    %
    properties
        Pair;                      % 真实的匹配
        FI;                        % 特征重要性
    end    
    methods
        function tasks = generateTask2(object,prm,proportion)
            Maxnum=prm.Maxnum;yuzhi=prm.yuzhi;dim=prm.dim;LEN=prm.LEN;T=prm.T;p=prm.p;F=prm.F;data=prm.data;GrtT=prm.GrtT;GrtR=prm.GrtR;
            

            SUM=LEN;
            l(1)=1;
            r(1)=SUM(1);
            for i=2:length(LEN)
                SUM(i)=SUM(i)+SUM(i-1);
                l(i)=SUM(i-1)+1;
                r(i)=SUM(i);
            end
            
            pa=zeros(1,dim);%promise area
            [~,idx]=sort(-object.FI);
            % pa(1,idx(1:100))=1;

            for i=1:length(LEN)
                [~,idx]=sort(-object.FI(l(i):r(i)));
%                 pa(l(i)-1+(idx(1:floor((1-LEN(i)/400)*LEN(i)))))=1;%特征数量越多，选的特征百分比就越小
%                 pa(l(i)-1+(idx(1:min(20,length(idx)))))=1;
%                 pa(l(i)-1+(idx(1:floor(0.2*LEN(i)))))=1;
%                 if i==1||i==4
%                     pa(l(i)-1+(idx(1:floor(0.6*LEN(i)))))=1;
%                 end
                if i==3||i==1
                    pa(l(i)-1+(idx(1:floor(proportion*LEN(i)))))=1;
                else
                    pa(l(i)-1+(idx(1:floor((1-LEN(i)/400)*LEN(i)))))=1;
                end
            end

            % pa(1,idx(1:200))=1;
            %仅为最重要的任务构造一个辅助特征
            X=zeros(1,dim);
            for i=1:1
                X(1,:)=pa;
%                 X(1,l(i):r(i))=1;
                tasks(1)=TASK();
                tasks(1)=initTASK(tasks(1),dim,X(1,:));
            end
            X=ones(1,dim);
            tasks(2)=TASK();
            tasks(2)=initTASK(tasks(2),dim,X(1,:));
        end
        
        function object = initDB(object,prm)
            Maxnum=prm.Maxnum;yuzhi=prm.yuzhi;dim=prm.dim;LEN=prm.LEN;T=prm.T;p=prm.p;F=prm.F;data=prm.data;GrtT=prm.GrtT;GrtR=prm.GrtR;
            
            SUM=LEN;
            l(1)=1;
            r(1)=SUM(1);
            for i=2:length(LEN)
                SUM(i)=SUM(i)+SUM(i-1);
                l(i)=SUM(i-1)+1;
                r(i)=SUM(i);
            end
            X=zeros(length(LEN),dim);
            for i=1:length(LEN)
                X(i,l(i):r(i))=1;
            end
            % X(i+1,:)=ones(1,dim);
            %这里计算FI和匹配点使用真实的匹配方法
            prm2=prm;
            prm2.p=prm2.pp;
            prm2.F=prm2.FF;
            object.Pair=generatePair(X,prm2);
            object=calcFI(object,prm);
        end  
        function object = calcFI(object,prm)
            Maxnum=prm.Maxnum;yuzhi=prm.yuzhi;dim=prm.dim;LEN=prm.LEN;T=prm.T;p=prm.pp;F=prm.FF;data=prm.data;GrtT=prm.GrtT;GrtR=prm.GrtR;
            pair=object.Pair;
            num=size(pair,1);%匹配的个数
            label=pair(:,5);
            PF=zeros(num,dim);
            
            for i=1:num
                f1=F{pair(i,1)}(pair(i,3),:);
                f2=F{pair(i,2)}(pair(i,4),:);
                PF(i,:)=rms(f1-f2,1)<yuzhi/5;
                PF(i,(f1==0)&(f2==0))=0;
%                 PF(i,:)=rms(f1-f2,1)<st/5;
            end
            object.FI=zeros(1,dim);
            for i=1:dim
%                 object.FI(:,i)=MutualInfo4(PF(:,i),label);
%                 object.FI(:,i)=rms(af(:,i))/10+MutualInfo4(PF(:,i),label);
%                 object.FI(:,i)=rms(af(:,i))*MutualInfo4(PF(:,i),label);
                object.FI(1,i)=abs(corr(PF(:,i),label));
            end
        end
        function tasks = generateTask(object,prm)
            Maxnum=prm.Maxnum;yuzhi=prm.yuzhi;dim=prm.dim;LEN=prm.LEN;T=prm.T;p=prm.p;F=prm.F;data=prm.data;GrtT=prm.GrtT;GrtR=prm.GrtR;
            

            SUM=LEN;
            l(1)=1;
            r(1)=SUM(1);
            for i=2:length(LEN)
                SUM(i)=SUM(i)+SUM(i-1);
                l(i)=SUM(i-1)+1;
                r(i)=SUM(i);
            end
            
            pa=zeros(1,dim);%promise area
            [~,idx]=sort(-object.FI);
            pa(1,:)=1;

            for i=1:length(LEN)
                [~,idx]=sort(-object.FI(l(i):r(i)));
%                 pa(l(i)-1+(idx(1:floor((1-LEN(i)/400)*LEN(i)))))=1;%特征数量越多，选的特征百分比就越小
%                 pa(l(i)-1+(idx(1:min(20,length(idx)))))=1;
%                 pa(l(i)-1+(idx(1:floor(0.2*LEN(i)))))=1;
%                 if i==1||i==4
%                     pa(l(i)-1+(idx(1:floor(0.6*LEN(i)))))=1;
%                 end
                if i==3||i==1
                    pa(l(i)-1+(idx(1:floor(0.9*LEN(i)))))=1;
                end
            end

%             pa(1,idx(1:200))=1;
            %仅为最重要的任务构造一个辅助特征
            X=zeros(1,dim);
            for i=1:1
                X(1,:)=pa;
%                 X(1,l(i):r(i))=1;
                tasks(1)=TASK();
                tasks(1)=initTASK(tasks(1),dim,X(1,:));
            end
            X=ones(1,dim);
            tasks(2)=TASK();
            tasks(2)=initTASK(tasks(2),dim,X(1,:));

%             %所有任务都构造一个特征，不是很有必要
%             X=zeros(size(LEN,1)+1,dim);
%             for i=1:length(LEN)
%                 X(i,:)=pa;
%                 X(i,l(i):r(i))=1;
%                 tasks(i)=TASK();
%                 tasks(i)=initTASK(tasks(i),dim,X(i,:));
%             end
%             i=i+1;
%             X(i,1:dim)=1;
%             tasks(i)=TASK();
%             tasks(i)=initTASK(tasks(i),dim,X(i,:));
        end
        function object = updateModel(object,idv,prm)
            pair=generatePair(idv,prm);%生成新的匹配
            object.Pair=[object.Pair;pair];
        end
    end
end