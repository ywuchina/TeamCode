function [LBEST,gbest,AD,W,Sample,M,pop,flag]=ppso1(pop,LBEST,popsize,k,X0,gbest,mm,u,W,Sample,M,AD_0,flag)
%--------�����Ӽ������۸���
% PopEff0=cell(M,1);
PopEff=cell(M,1);
%  %----------------------------------
for I=1:M
    PopEff{I}=evaluation(pop,popsize,mm,Sample{I},u,k);%ÿ��Ԫ���ĵ�mm+1���ǵ�M�������Ӽ��ϵķ��ྫ�ȣ���������Ԫ��mm+2�н��һ�������Ӱ���������Ŀ
end
% ------����Ȩ�ؼ���������Ӧֵ

popeff=zeros(popsize,mm+3+M);%mm+3�����ӵĲ�ȷ���ȣ�mm+1�ż�Ȩ�������Ӧֵ��mm+3+1��mm+3+M���M������Ӧֵ��
popeff(:,1:mm)=pop;
%-------�����ӵļ�Ȩ��Ľ�����Ӧֵ�Ͳ�ȷ����
for i=1:popsize
    ary=0;
    av=0;%������Ӧֵ��ֵ
    Uncertainty=0;%��ȷ����
    sub_eff=[];%��������ÿ��������ÿ�������Ӽ��ϵ���Ӧֵ
    for j=1:M
        a_f=PopEff{j}(i,mm+1);
        ary=ary+W(j)*a_f;
        av=av+a_f;
        sub_eff=[sub_eff,a_f];%������ʽ��һ��M�С�
    end
    av=av/M;
    for j=1:M
        a_f=PopEff{j}(i,mm+1);
        Uncertainty=Uncertainty+(a_f-av)^2;
    end
    Uncertainty=(Uncertainty/(M-1))^0.5;
    
    popeff(i,mm+1)=ary;%��Ȩ����Ӧֵ
    popeff(i,mm+2)= PopEff{1}(i,mm+2);
    popeff(i,mm+3)=Uncertainty;
    popeff(i,mm+4:end)=sub_eff;%��mm+4�д�ŵ�һ�������Ӽ��ϵ���Ӧֵ
    
end
%------------���弫ֵ��ĸ���(���ڼ�Ȩ����Ӧֵ�Ļ�����)

for i=1:popsize
    if popeff(i,mm+1)>LBEST(i,mm+1) 
        LBEST(i,1:mm+2+M)=[popeff(i,1:mm+2),popeff(i,mm+4:end)];
    elseif popeff(i,mm+1)==LBEST(i,mm+1)&&popeff(1,mm+2)<LBEST(i,mm+2)
        LBEST(i,1:mm+2+M)=[popeff(i,1:mm+2),popeff(i,mm+4:end)];
    end
end

%------------  ȫ�ּ�ֵ��ĸ���
[~,index]=max(LBEST(:,mm+1));
best=LBEST(index,1:mm+2);%�ڽ�����Ӧֵ�����best�ĵ�mm+1λ�Ž�����Ӧֵ
%--------������ʵ��Ӧֵ
[aa,~]=kmean_res(best(1:mm),X0,u,k,mm);
zhen_best=best;%�ڽ�����Ӧֵ�����zhen_best�ĵ�mm+1λ�ż�Ȩ�������Ӧֵ��zhen_best�ĵ�mm+2λ��������ģ
zhen_best(mm+1)=aa*100;%zhen_best�ĵ�mm+1λ�ŵ�����ʵ��Ӧֵ
if zhen_best(1,mm+1)>gbest(1,mm+1)
    gbest(1,1:mm+2)=zhen_best(1,1:mm+2);%%%%%-----���������������У� gbest��Сβ�Ͷ�����ʵ��Ӧֵ
    %%-----��Ȩ��Ӧֵ��Ϊ�˺����ж�ģ���Ƿ����
    gbest(1,mm+3)=best(1,mm+1);
    
elseif zhen_best(1,mm+1)==gbest(1,mm+1)&&zhen_best(1,mm+2)<gbest(1,mm+2)
    gbest(1,1:mm+2)=zhen_best(1,1:mm+2);
    %%-----��Ȩ��Ӧֵ��Ϊ�˺����ж�ģ���Ƿ����
    gbest(1,mm+3)=best(1,mm+1);
end


%------------------------- ���ε���ͨ�������㣬
%-------Ȩ�ظ��´�����AD����䣺1�������Ž⣬2��ȷ��������
%---------Ȩ�ظ��´�����,1:mm�з����ӣ�mm+1�ż�Ȩ����Ӧֵ��mm+2��������Ŀ,mm+3����ʵ��Ӧֵ
DD=[];
%--������Ӧֵ��������---BBest:mm+1�ǽ�����Ӧֵ��mm+2��������ģ��mm+3����ʵ��Ӧֵ

BBest=[best,aa*100];%������Ӧֵ��������---BBest:mm+1�ǽ�����Ӧֵ��mm+2��������ģ��mm+3����ʵ��Ӧֵ

%--��ȷ������������
[~,index]=max(popeff(:,mm+3));
UBest=popeff(index,1:mm+2);%��mm+1�Ǽ�Ȩ�н�����Ӧֵ
%--------������ʵ��Ӧֵ
[b,~]=kmean_res(UBest(1:mm),X0,u,k,mm);
UBest=[UBest,b*100];%mm+3����ʵ��Ӧֵ

%-----���ε�����BBest��UBest����Ȩ�ظ��´�����
DD=[BBest;UBest];
%---------------------------------
    %-------ȷ�����ɴ���Ȩ�ظ���ʱ��
     if flag==5 || flag>5
            %------�����仯����Ҫ���¸��³�ʼ��Ⱥpop������Ȩ��W�� ���µ�ǰLBEST,gbest,����AD
            M_0=1;
            M=M-round(M_0);
            kk=size(X0,1);
            Sample=Sample_grouping(kk,X0,M);
            %----------------------------------
            %-------���µ�ǰ��Ⱥ
            %-------һ������Ҫ����
            %---------�ҵ���һʱ��Ȩ���£�LBEST��Ȩ����Ӧֵ��С��5%�ĸ�����������ʼ����
            popsize1=ceil(0.05*popsize);
            pop1=zeros(popsize1,mm);
            for i=1:(popsize1)
                for j=1:mm
                    pop1(i,j)=randperm(length(u{j})+1,1)-1;
                    %-----�͵�ǰ���Ž⽻�����
                    pop1(i,j)=round(0.5*(pop1(i,j)+gbest(1,j)));
                end
            end
            [~,index]=sort(LBEST(:,mm+1));%��������
            set=index(1:popsize1);
            LBEST(set,:)=[];
            pop=[pop1;LBEST(:,1:mm)];
            %---------------------------------------
            %--------�µ���Ⱥ�������Ӽ������½������۵�ǰ�����ĳ�ʼ��Ⱥ
            sub_popeff=cell(M,1); %��Ԫ����������M�������Ӽ��ϵ���Ⱥ���ྫ��
            pop_0=pop(:,1:mm);%��ʼ��Ⱥ
            
            %-----------------------
            pop=pop_0;
            % parpool 'local' 'MATLAB Parallel Cloud';
            for i=1:M
                sub_popeff{i}=evaluation(pop_0,popsize,mm,Sample{i},u,k);%��ǰ�����Ӽ������£���Ⱥ��ÿ�������Ӽ��µ���Ӧֵ
            end
            % parpool closed;
            %----------------------------------

            %-------���¸���Ȩ��
            %-------(���ַ�ʽ��Ҫ���¼��㵱ǰ����Ⱥpop����ʵ��Ӧֵ����ʱ)���¸��´�������=��ǰ����Ⱥpop+DD(��������+��ȷ�������)+DB(ÿ�������Ӽ��µ���������)
            %-------���¸��´�������=��һ����ʼ��Ⱥpop(��ʵ��Ӧֵ�Ѿ������㣬ʡʱ)+DD(��������+��ȷ�������)+DB(ÿ�������Ӽ��µ���������)
            AD=[AD_0(1:popsize,:);DD];
            %-----����Ҫ���������������ϻ�����ÿ�������Ӽ��µ���������
            DB=[];

            for s=1:M
                pp=sub_popeff{s};
                [~,index1]=max(pp(:,mm+1));
                Sub_best=pp(index1,1:mm+2);%�ڽ�����Ӧֵ�����Sub_best�ĵ�mm+1λ������һ�����������½�����Ӧֵ����Ҫ���µ����������޸�
                [sb,~]=kmean_res(Sub_best(1:mm),X0,u,k,mm);%Ҫ����ʵ����
                Sub_best=[Sub_best,sb*100];%mm+3����ʵ��Ӧֵ
                DB=[DB;Sub_best];
            end
            AD=[AD;DB];
            %---------------------------------------

            %----�������»��ֺ����Ȩ��
            popeff_0=[AD(:,1:mm),AD(:,mm+3),AD(:,mm+2)];
            ppsize=size(popeff_0,1);
            W=sub_model_1(popeff_0,Sample,k,M,mm,u,ppsize);%----��������ʱ��ͬʱ����W��
            %------����Ȩ�ؼ���������Ӧֵ
            w_eff=zeros(popsize,mm+2+M);%mm+1�ż�Ȩ�������Ӧֵ
            w_eff(:,1:mm)=pop_0;
            %-------�����ӵļ�Ȩ��Ľ�����Ӧֵ,�Լ���ÿ�������Ӽ��ϵ���Ӧֵ
            for i=1:popsize
                ary=0;
                sub_eff=[];%��������ÿ��������ÿ�������Ӽ��ϵ���Ӧֵ
                for j=1:M
                    a_f=sub_popeff{j}(i,mm+1);
                    sub_eff=[sub_eff,a_f];%������ʽ��һ��M�С�
                    ary=ary+W(j)*a_f;
                end

                w_eff(i,mm+1)=ary;
                w_eff(i,mm+2)= sub_popeff{1}(i,mm+2);
                w_eff(i,mm+3:end)=sub_eff;%��mm+3�д�ŵ�һ�������Ӽ��ϵ���Ӧֵ
            end

            %---------��ʼ��ֵ��ĸ���
            LBEST=w_eff;%���治������ʵ��Ӧֵ��mm+1��Ȩ��Ӧֵ��mm+2�����Ӽ���ģ����mm+3�д�ŵ�һ�������Ӽ��ϵ���Ӧֵ
            %--------�µ���Ⱥ�������Ӽ�����������gbest
            %------------  ȫ�ּ�ֵ��ĸ���
            [~,index]=max(LBEST(:,mm+1));
            best=LBEST(index,1:mm+2);%�ڽ�����Ӧֵ�����best�ĵ�mm+1λ�Ž�����Ӧֵ
            %--------������ʵ��Ӧֵ
            [aa,~]=kmean_res(best(1:mm),X0,u,k,mm);
            zhen_best=best;%�ڽ�����Ӧֵ�����zhen_best�ĵ�mm+1λ�ż�Ȩ�������Ӧֵ��zhen_best�ĵ�mm+2λ��������ģ
            zhen_best(mm+1)=aa*100;%zhen_best�ĵ�mm+1λ�ŵ�����ʵ��Ӧֵ
            if zhen_best(1,mm+1)>gbest(1,mm+1)
                gbest(1,1:mm+2)=zhen_best(1,1:mm+2);%%%%%-----���������������У� gbest��Сβ�Ͷ�����ʵ��Ӧֵ
                %%-----��Ȩ��Ӧֵ��Ϊ�˺����ж�ģ���Ƿ����
                gbest(1,mm+3)=best(1,mm+1);

            elseif zhen_best(1,mm+1)==gbest(1,mm+1)&&zhen_best(1,mm+2)<gbest(1,mm+2)
                gbest(1,1:mm+2)=zhen_best(1,1:mm+2);
                %%-----��Ȩ��Ӧֵ��Ϊ�˺����ж�ģ���Ƿ����
                gbest(1,mm+3)=best(1,mm+1);
            end
            %---------------------------------------------------
            %-------�����仯��������flag
            flag=0;
            
            
            
            %               flag=0;G_feal=gbest(mm+1);
            %               G_feal=gbest(mm+1);
     else
            %------�����һ�����»��������ˣ���ôȨ�ؾ��Ѿ������¸�����,����Ҫ�ظ�ִ��Ȩ�ظ��»���
            %------û�����»����������ж��Ƿ����Ȩ��
            %------������Ȩ�ظ���ʱ�̵�ȫ�ּ�ֵ��͸��弫ֵ��
            AD=[AD_0;DD];%----���������
            %------�жϴ���ģ�����
            E=abs(gbest(mm+3)-gbest(mm+1))/(gbest(mm+1));
            if E>0.01  
                %-------����Ȩ��(ע�⣺�˿̵�AD_0���Ѿ����Ӻ���ppso1.m�б����¹������ظ�����)
                popeff_0=[AD(:,1:mm),AD(:,mm+3),AD(:,mm+2)];
                ppsize=size(popeff_0,1);
                W=sub_model_1(popeff_0,Sample,k,M,mm,u,ppsize);
                %---------------------------------------------

                %--------�µ�Ȩ���¸��¼�Ȩ��Ӧֵ�����������Ӽ�û�����»��֣�ֻ��Ҫ�����µ�Ȩ�ؼ����Ȩ����Ӧֵ,
                for i=1:popsize
                    ary=0;
             
                    for j=1:M
                        a_f=LBEST(i,mm+2+j);
                        ary=ary+W(j)*a_f;
                    end
                    
                    LBEST(i,mm+1)=ary;%---------��mm+1���Ǽ�Ȩ��Ӧֵ��
                   
                end
                %-------------------------------------------------------

            end
            %-------------------------------------
     end


       