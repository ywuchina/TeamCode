function [best,time] = nomultitask(prm,d)
%NOMULTITASK 此处显示有关此函数的摘要
%   此处显示详细说明
    tic
    Maxnum=prm.Maxnum;yuzhi=prm.yuzhi;dim=prm.dim;LEN=prm.LEN;T=prm.T;p=prm.p;F=prm.F;data=prm.data;GrtT=prm.GrtT;GrtR=prm.GrtR;
    
    tasks=TASK();
    tasks=initTASK(tasks,prm.dim,ones(1,prm.dim));
    tn=length(tasks);    %task number 任务个数

%     load("data\PC\PF&Label");
%     load("data\PC\I.mat");
    TE=1;%真实评估次数
    te=0;
    ite=10;%迭代数
    pop=10;%个体数
    pc=0.1;%杂交概率
    pm=20/dim;%变异概率
    elite_num=1;
%     arr=(1-LEN/400);
%     arr(3)=0.9;
%     arr1=ones(1,4);
    arr1=[0.5,0.5,0.5,0.5];
    
    W1=zeros(1,dim);
    t=1;
    for i=1:length(LEN)
        W1(t:t+LEN(i)-1)=arr1(i);
        t=t+LEN(i);
    end
    idvs(1)=INDIVIDUAL();
    idvs(1)=initINDIVIDUAL(idvs(1),prm,pop,tasks(1),pc,pm,elite_num,W1);
%     for i=1:tn
%         idvs(i)=INDIVIDUAL();
%         idvs(i)=initINDIVIDUAL(idvs(i),prm,pop,tasks(i),pc,pm,elite_num,W);
%     end
    fitss=zeros(tn,10000);
    truefits=zeros(tn,10000);
    ppp=zeros(tn*2,1);
    prm2=prm;
    prm2.p=prm2.pp;
    prm2.F=prm2.FF;

%     prm=prm2;

    while(te<TE)
        for it=1:ite
            pairs=generatePairs(idvs);
            o1=cross(idvs,pairs);
            o2=mutation(idvs,o1);
            
            all=[idvs.idvs;o1;o2];
            nnfits=eva4([o1;o2],prm);
            nfits=[idvs.fits;nnfits];
            for i=1:tn
                bestfits(i,te*ite+it)=idvs(i).fits(1);
                nums(i,te*ite+it)=sum(idvs(i).idvs(1,:));
%                 disp(i+": "+idvs(i).fits(1));
            end
            

            
        end
        te=te+1;
        if te<TE
            d=updateModel(d,idvs(tn).idvs(1,:),prm2);
            d=updateModel(d,idvs(tn).idvs(2,:),prm2);
            d=updateModel(d,idvs(1).idvs(1,:),prm2);
            d=updateModel(d,idvs(1).idvs(2,:),prm2);
            calcFI(d,prm);
    %         disp("第"+te+"次迭代特征重要性：");
    %         disp(d.FI);
            tasks=generateTask(d,prm);
            for j=1:tn
                if j==1
                    idvs(j)=generateNewIndividualFromOld(idvs(j),tasks(j),W1);
                else
                    idvs(j)=generateNewIndividualFromOld(idvs(j),tasks(j),W2);
                end
                
            end
        end
        
        
    end
    for i=1:tn
%         tf=zeros(1,5);
%         for j=1:5
%             X=idvs(i).idvs(j,:);
%             tf(j)=eva4(X,prm2);
%         end
%         [~,idxx]=max(tf);
%         best(i,:)=idvs(i).idvs(idxx,:);
        best(i,:)=idvs(i).idvs(1,:);
    end
    best=best(1,:);
    disp("本方法运行时间：");
    time=toc;

    disp(idvs(tn).fits(1));
    l=1;r=33;
    s="";
    for i=1:4
        s=s+sum(best(1,l:r))+" ";
        if i~=4
            l=r+1;
            r=l+LEN(i+1)-1;
        end
    end
    disp(s);


%     figure
%     plot(1:(ite*TE),bestfits);
%     title("匹配正确性进化过程");
%     figure
%     plot(1:(ite*TE),nums);
%     title("最好个体维度变化");
%     for i=1:tn
%         disp(sum(best(i,:)));
%     end
%      for i=1:1
%         fits1=fitss(i,1:ppp(i));
%         fits2=truefits(i,1:ppp(i+tn));
%         figure
%         [~,idx]=sort(fits1);
% %         tfits1=fits1-(sum(fits1)/length(fits1)-sum(fits2)/length(fits2));
%         plot(1:length(idx),fits1(idx));
%         hold on
%         plot(1:length(idx),fits2(idx));
% 
%         %拟合曲线
%         x=1:length(idx);
%         y1=fits2(idx);       
%         P=polyfit(x,y1,3);
%         y2=polyval(P,x);
%         plot(x,y2);

%         legend('代理函数适应度','真实适应度');
%         xlabel('迭代次数');
%         ylabel('适应度值');
%         title("第"+i+"个任务真实适应度与代理适应度关系(大小顺序)");

%         figure
%         tfits1=fits1-(sum(fits1)/length(fits1)-sum(fits2)/length(fits2));
%         plot(1:length(idx),fits1);
%         hold on
%         plot(1:length(idx),fits2);
%         legend('代理函数适应度','真实适应度');
%         xlabel('迭代次数');
%         ylabel('适应度值');
%         title("第"+i+"个任务真实适应度与代理适应度关系(迭代顺序)");
%     end
end

