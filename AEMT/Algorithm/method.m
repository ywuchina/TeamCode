function [best,time] = method(prm,d)
%根据输入的数据选择出最优的特征子集
%   prm是参数
    tic
    Maxnum=prm.Maxnum;yuzhi=prm.yuzhi;dim=prm.dim;LEN=prm.LEN;T=prm.T;p=prm.p;F=prm.F;data=prm.data;GrtT=prm.GrtT;GrtR=prm.GrtR;
    
    
    
    tasks=generateTask(d,prm);
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
    arr1=[0.8,0.5,1,0.5];
    arr2=[0.8,0.5,1,0.5];
    W1=zeros(1,dim);
    W2=zeros(1,dim);
    t=1;
    for i=1:length(LEN)
        W1(t:t+LEN(i)-1)=arr1(i);
        W2(t:t+LEN(i)-1)=arr2(i);
        t=t+LEN(i);
    end

    idvs(1)=INDIVIDUAL();
    idvs(1)=initINDIVIDUAL(idvs(1),prm,pop,tasks(1),pc,pm,elite_num,W1);
    idvs(2)=INDIVIDUAL();
    idvs(2)=initINDIVIDUAL(idvs(2),prm,pop,tasks(2),pc,pm,elite_num,W2);
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

            for j=1:tn %杂交变异
                pairs=generatePairs(idvs(j));
                o1{j}=cross(idvs(j),pairs);
                o2{j}=mutation(idvs(j),o1{j});
            end

            for j=1:tn %知识迁移
                if j==tn%对于全部特征空间，其他的任务都选择最好的个体转移过去
                    to3=[];
                    to4=[];
                    for z=1:tn-1
                        to=idvs(z).idvs(1,:);%直接将个体转移过去
                        to3=[to3;to];
                        to=idvs(j).idvs(1,:);
                        to(idvs(z).task.space==1)=idvs(z).idvs(1,idvs(z).task.space==1);%将个体的一段替换一个个体的一段
                        to3=[to3;to];

                    end
                else
                    to3=zeros(1,dim);
                    to3(idvs(j).task.space==1)=idvs(tn).idvs(1,idvs(j).task.space==1);%全局最优
                end
                o3{j}=to3;
%                 o3{j}=[];%不知识传递
            end
            
            for j=1:tn %评估新个体 选择
                all=[idvs(j).idvs;o1{j};o2{j};o3{j}];
                nnfits=eva4([o1{j};o2{j};o3{j}],prm);
                nfits=[idvs(j).fits;nnfits];

                [idv,fits]=select(idvs(j),all,nfits);
                idvs(j).idvs=idv;
                idvs(j).fits=fits;
            end
            
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
    % disp("本方法运行时间：");
    time=toc;

    % disp(idvs(tn).fits(1));
    l=1;r=33;
    s="";
    for i=1:4
        s=s+sum(best(1,l:r))+" ";
        if i~=4
            l=r+1;
            r=l+LEN(i+1)-1;
        end
    end
    % disp(s);
    
    
end
function [idvs,fits]=localsearch(idvs,fits,allF,pairs,label,W,yuzhi)
    for i=1:size(idvs,1)
        r=floor(rand(1,3)*size(idvs,2))+1;
        [~,idx]=sort(r);
        loc1=r(idx(1));
        loc2=r(idx(2));
        loc3=r(idx(3));
        nidv=idvs(i,:);
        t1=nidv(loc1:loc2-1);
        t2=nidv(loc2:loc3);
        nidv(loc1:(loc3-loc2)+loc1)=t2;
        nidv((loc3-loc2+1)+loc1:loc3)=t1;
        y=eva3(p,nidv,allF,pairs,label,W,yuzhi);
        if fits(i)<y
            idvs(i,:)=nidv;
            fits(i)=y;
        end
    end
end
function dopicture(PF,Ix,Ixy)
%做两个评估函数的图
    num=2^8;
    for i=1:num
        t=i;
        xidv=zeros(1,12);
        p=1;
        while t>0
            if mod(t,2)==1
                xidv(p)=1;
            end
            p=p+1;
            t=floor(t/2);
        end
        for j=1:num
            t=j;
            yidv=zeros(1,12);
            p=1;
            while t>0
                if mod(t,2)==1
                    yidv(p)=1;
                end
                p=p+1;
                t=floor(t/2);
            end
            idvs(i,j).chro=[xidv,yidv,ones(1,513)];
        end
    end
    for i=1:num
        for j=1:num
            mrmrfits(i,j)=eva(idvs(i,j).chro,PF,Ix,Ixy);
            mrfits(i,j)=eva2(idvs(i,j).chro,PF,Ix,Ixy);
        end
    end
    mm=mean(mrmrfits)-mean(mrfits);
    mrmrfits=mrmrfits-mm;
    surf(1:num,1:num,mrmrfits,'EdgeColor','r');
    hold on;
    surf(1:num,1:num,mrfits,'EdgeColor','b');
end
