function [best_member, F1_new] = multitask(X,Y,prm)
    % parameter setting
    n_feature = size(X, 2);
    pool=ones(1,n_feature);
%     n_obj = 3;
    group=1;
    task_size=[];
%     n_pop = min(round(n_feature / 20), 300); 
%     n_pop = 30;
    n_iter = 1;
    mul=zeros(1,n_feature);
       [idx, weight] = relieff(X,Y,1);%weigth是1~n每个特征为的权重，idx是根据weigth排序后的结果。

      for i = 1:n_feature
         %X1=X(:,i);
         mul(i) = MItest(X(:,i),Y);
      end
       [mul_rank,pos]=sort(mul);
       mul_rank1=fliplr(mul_rank);
       pos1=fliplr(pos);
%        plot(mul_rank1);
       fea_rank=[pos1',mul_rank1'];

num_data=size(X,1);
TV = zeros(1,n_feature);
for d = 1:n_feature

  mu    = mean(X(:,d));

  TV(d) = (1 / num_data) * sum((X(:,d) - mu) .^ 2);
end
[~, idx] = sort(TV,'descend');
weight1=TV;

X_mu      = mean(X,1); 
Y_mu      = mean(Y,1); 
dX       = X - X_mu; 
dY       = Y - Y_mu;
nume     = sum((dX .* dY), 1); 
deno     = sqrt(sum(dX .^ 2, 1) .* sum(dY .^2, 1));
pcc      = nume ./ deno; 
pcc      = abs(pcc);
[~, idx] = sort(pcc,'descend');
weight2=pcc;

% TV=sort(TV);
% plot(TV);
%     
     %divide group
          [task_map,task_size,pool]=divide_four_task(X,n_feature,idx,weight,weight1,weight2,pool);
       
     task_num=4 ;
     
     
     n_pop = 4*task_num;
    individual.taskfea=[];
    individual.position = [];
    individual.cost = [];
    individual.task_mark=[];
    lastp.position=[];
    lastp.cost=[];
    pop = repmat(individual, n_pop, 1);
    
    j=0;
     for i = 1:n_pop
        ii=mod(i,task_num);
        if ii==1
            j=j+1;
        end
        if ii==0
            ii=task_num;
        end
        position = rand(1,n_feature);
        pop(i).taskfea=task_map(ii,:);
        pop(i).position = position;
        for q=1:n_feature
            if (pop(i).position(q)>0.5)&&(pop(i).taskfea(q)==1)
                
            else
                pop(i).position(q)=0;
            end
        end
        pop(i).cost=eva4(pop(i).taskfea,prm);
        pop(i).task_mark=ii;
        task_pop(ii,j)=i;
     end
     
for iter = 1:n_iter
    iter_start = clock;
    winner_all=[];
    loser_all=[];
    v=zeros(n_pop,n_feature);
    for task_t=1:task_num
    winner=[];
    loser=[];
    task_in=task_pop(task_t,:);
    p_num=size(task_in,2);
    p=randperm(size(task_in,2));
    p_new=task_in(:,p);
    run_time=1;
   while(run_time~=(p_num+1))
%         a=pop(p_new(run_time));
%         b=pop(p_new(run_time+1));
        [win,lose]=compare(pop,p_new,run_time);
        winner=[winner;win];
        loser=[loser;lose];
        run_time=run_time+2;
    end
    winner_all=[winner_all;winner];
    loser_all=[loser_all;loser];
    end
    transp=0.5;% 迁移概率p=0.5
    loser_num=size(loser_all,1);

    for learn_time=1:loser_num
        if transp<rand

            win=winner_all(learn_time);
            lose=loser_all(learn_time);
            win2=win;
            loser_mark=pop(lose).task_mark;
%             if pop(lose).task_mark==4
                mark=zeros(4,1);
                trans_win=zeros(4,1);
                mark(loser_mark,1)=1;
%                 mark(4,1)=1;
%                 trans_win(4,1)=win;
                trans_win(loser_mark,1)=win;
                while (pop(win).task_mark==pop(lose).task_mark)||(sum(mark)<4)
                rand_win=randperm(size(winner_all,1));
                win1=winner_all(rand_win(1),:);
                win_task=pop(win1).task_mark;
                if (pop(win1).task_mark~=pop(lose).task_mark)&&(mark(win_task)==0)
                    win=win1;
                    t_task=pop(win1).task_mark;
                    mark(t_task,1)=1;
                    trans_win(win_task,1)=win;
                end
                end
                if loser_mark==4
                    combine_win=0.45*pop(trans_win(1)).position+0.45*pop(trans_win(2)).position+0.45*pop(trans_win(3)).position;
                else
                    combine_win=0.45*pop(trans_win(1)).position+0.45*pop(trans_win(2)).position+0.45*pop(trans_win(3)).position+0.1*pop(trans_win(4)).position-0.45*pop(trans_win(loser_mark)).position;
                end
            %combine_win=0.3*pop(trans_win(1)).position+0.3*pop(trans_win(2)).position+0.3*pop(trans_win(3)).position-0.3*pop(trans_win(loser_mark)).position;
            new_v=rand*v(win,:)+rand*(combine_win-pop(lose).position)+rand*(pop(win2).position-pop(lose).position);
%             else
%                 while (pop(win).task_mark==pop(lose).task_mark)&&(pop(win).cost(1)<=pop(lose).cost(1))
%                 rand_win=randperm(size(winner_all,1));
%                 win=winner_all(rand_win(1),:);
%                 end 
%                 new_v=rand*v(win,:)+rand*(pop(win).position-pop(lose).position)+rand*(pop(win2).position-pop(lose).position);
%             end
            loser_position=pop(lose).position+new_v+rand*(new_v-v(lose,:));
            v(lose,:)=new_v;
            pop(lose).position=loser_position;
        else
            win=winner_all(learn_time);
            lose=loser_all(learn_time);
            new_v=rand*v(win,:)+rand*(pop(win).position-pop(lose).position);
            loser_position=pop(lose).position+new_v+rand*(new_v-v(lose,:));
            v(lose,:)=new_v;
            pop(lose).position=loser_position;
        end
        pop(lose).position=jiaozheng(pop(lose).position,n_feature);
        pop(lose).cost=eva4(pop(lose).taskfea,prm);
    end
     [winner_offspring,pop]=PM(winner_all,n_feature,X,Y,pop,prm);
        % analysis
        iter_end = clock;
        avgfit=mean([pop.cost]);
        % logger(['iter: ', num2str(iter), '/', num2str(n_iter), ' time: ', num2str(etime(iter_end, iter_start)), 's', ...
        %     ' fit: ', num2str(avgfit(1))]);
end
F1_num=size(pop,1);
 F1_new=repmat(lastp,F1_num,1);
 for aa=1:F1_num
      F1_new(aa).position = unifrnd(0, 1, 1, n_feature) > 0.5;
     for j=1:n_feature     
        if (pop(aa).position(j)>0.5)
            F1_new(aa).position(j)=1;
        else
            F1_new(aa).position(j)=0;
        end
     end
     if sum(F1_new(aa).position)==0
         F1_new(aa).position(1)=true;
     end
     F1_new(aa).cost=eva4(F1_new(aa).position,prm);
     if aa==1
         best_member=F1_new(aa);
     end
     if F1_new(aa).cost<best_member.cost
         best_member=F1_new(aa);
     end
 end
end

function best_member = solution_selection(F1)
    n_F1 = numel(F1);
    best_member = F1(1);
    for i = 2:n_F1
        if F1(i).cost(1) < best_member.cost(1)
            best_member = F1(i);
        elseif F1(i).cost(1) == best_member.cost(1) && F1(i).cost(2) < best_member.cost(2)
            best_member = F1(i);
        elseif F1(i).cost(1) == best_member.cost(1) && F1(i).cost(2) == best_member.cost(2)&&F1(i).cost(3) < best_member.cost(3)
            best_member = F1(i);
        end
    end
end
function [childern,pop] = PM(winner,n_feature,X,Y,pop,prm)
     parent=pop(winner);
    N=size(parent,1);
    D=n_feature;
    individual.taskfea=[];
    individual.position = [];
    individual.cost = [];
    individual.task_mark=[];
    childern = repmat(individual, N, 1);
    disM=20;
    for i=1:N
        Offspring(i,:)=parent(i).position;
    end
     Lower = repmat(0,N,D);
     Upper = repmat(1,N,D);
     Site  = rand(N,D) < 1/D;
     mu    = rand(N,D);
     temp  = Site & mu<=0.5;
     Offspring       = min(max(Offspring,Lower),Upper);
     Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                              (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
     temp = Site & mu>0.5; 
     Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
     (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
     for i=1:N
         childern(i).taskfea=parent(i).taskfea;
         childern(i).position=Offspring(i,:);
         childern(i).position=jiaozheng(childern(i).position,n_feature);
          childern(i).task_mark=parent(i).task_mark;
          childern(i).cost = eva4(childern(i).taskfea,prm);
          if childern(i).cost(1)>parent(i).cost(1)
             pop(winner(i)).position=parent(i).position;
             pop(winner(i)).cost=parent(i).cost;
          else
              pop(winner(i)).position=childern(i).position;
             pop(winner(i)).cost=childern(i).cost;
          end
     end

end