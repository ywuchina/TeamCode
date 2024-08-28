function [best_member, F1] = MOEAD(X, Y,k)
    % parameter setting
    n_feature = size(X, 2);
    n_obj = 3;
%     n_pop = min(round(n_feature / 20), 300); 
    n_pop = 300;
    n_iter = 70;
    mul=zeros(1,n_feature);
    pi1=[];
    pi2=[];
    for tt=1:n_iter
%         a=(1-0.15)*(1+0.15*exp(8*(tt/n_iter-0.35))).^(-1);
         a=(1-0.15)*(1+0.15*exp(10*(tt/n_iter-0.6))).^(-1);
        pi1=[pi1,a];
%         b=0.5*abs(sin(0.045*(tt-0.5))+0.3);
%          b=1/(pi*0.6*(1+((tt-35)/15)^2))+0.2;
%          b=1/(pi*0.5*(1+((tt-35)/25)^2))+0.05;
%         pi2=[pi2,b];
    end
%          plot(pi2);
        pi3=pi2(randperm(size(pi2,2)));
     %   plot(pi3);

%       plot(pi);
%       xlabel('{\itt}');
%       ylabel('{\itp}{_1}');
%     Z=[];
      for i = 1:n_feature
         %X1=X(:,i);
         mul(i) = MItest(X(:,i),Y);
      end
       pmin=min(mul);
       
     %  mul_rank=-sort(-mul);
%       mul_rate=mul/max(mul);
       mul_rate1=(1-pmin)*(1+pmin*exp(10*(mul-0.3))).^(-1);
       
        mul_rate2=mul/max(mul);
        mul_rank=-sort(-mul_rate1);
%          plot(mul_rank);
  %生成权重向量
    [W,n_pop]=UniformPoint(n_pop,n_obj);
    T = ceil(n_pop/10);
    
     % Detect the neighbours of each solution
    B = pdist2(W,W);
    [~,B] = sort(B,2);
    B = B(:,1:T);
    
%    % generate reference points
%     zr = reference_points(n_obj, n_division);

    % initialization
    individual.position = [];
    individual.cost = [];
    pop = repmat(individual, n_pop, 1);
    Cheb=ones(n_pop,1);%聚合函数值
    for i = 1:n_pop
        position = unifrnd(0, 1, 1, n_feature) > 0.5;
        pop(i).position = position;
        pop(i).cost = fitness(X, Y, position);
        deta(i)=0.35;
    end
      Z = min([pop.cost]', [], 1);
      Zmax  = max( [pop.cost]', [], 1);
      Cheb_old=ones(n_pop,1);
%        for j=1:n_pop
%                Parents = B(j,randperm(size(B,2)));
%               pp=pop(Parents).cost;
%               pp=pp';
%               g= max(abs(pp-repmat(Z,T,1))./repmat(Zmax-Z,T,1).*W(Parents,:),[],2);
% %                Cheb(j)=g;
%             %  Cheb(Parents)=g; 
%                for jj=1:size(Parents)
%                     if(g(jj)<Cheb(Parents(jj)))
%                         Cheb(Parents(jj))=g(jj);
%                     end
%                end
%              
%        end
        pp=pop.cost;
        Cheb= max(abs(((pp)'-repmat(Z,n_pop,1)).*W),[],2);
%         deta1=abs((Cheb_old-Cheb));
%         deta=deta1/max(deta1)*0.5+0.3;
%         plot(deta);
%        deta=0.35;
       Cheb2=Cheb*(1-mul);
       [Cheb_rank,I]=sort(Cheb);
       elites_size=ceil(0.05*n_pop);
       elites=pop(I(1:elites_size));
       
       p1=0.5;
        F=nondominated_sort(pop);
          F1=pop(F{1});
          Fl = pop(F{end});
    % run iterations
    for iter = 1:n_iter
        Cheb_old=Cheb;
        iter_start = clock;
        for i=1:n_pop
             % Choose the parents
            Parents = B(i,randperm(size(B,2)));
             if deta(iter)<rand
%            if 0.65>rand
               % Generate an offspring
             popc = crossover_pop(pop(Parents(1:2)),1,X,Y);
   %           Offspring=mutate_pop(popc,1, X, Y,Fl);
             pp=pi1(iter);
              popm=mutate(popc,n_feature, X, Y,mul_rate1,mul_rate2,elites,pp);
%               %repair
                    p=randperm(size(elites,1));
                    elite=elites(p(1));
                    gel=Cheb2(I(p(1)),:);
                    of_g=min(max(repmat(abs((popm.cost)'-Z)./(Zmax-Z),T,1).*W(Parents,:),[],2));
                    gof=of_g*(1-mul);
                   p2=mul/max(mul);
                   Offspring=repair(popm,elite,n_feature,gel,gof, X, Y,p1,p2);  
            else
                Offspring=creatDE(pop(Parents(1:4)),n_feature,X,Y,iter);
            end
            
              % Update the ideal point
               Z = min(Z, [Offspring.cost]');
              % Tchebycheff approach with normalization
             Zmax  = max( [pop.cost]', [], 1);
             pp=zeros(size(Parents,2),n_obj);
             for kk=1:size(Parents,2)
%                  pp=[pp;pop(Parents(i)).cost];
                 pp(kk,:) = pop(Parents(kk)).cost;
             end
              
%               pp=pp';
              g_old = max(abs(pp-repmat(Z,T,1))./repmat(Zmax-Z,T,1).*W(Parents,:),[],2);
              g_new = max(repmat(abs((Offspring.cost)'-Z)./(Zmax-Z),T,1).*W(Parents,:),[],2);
              
              pop(Parents(g_old>=g_new)) = Offspring;
%               for jjj=1:size(g_new,1)
%                   if Cheb(Parents(jjj))>=g_new
%                       Cheb(Parents(jjj))=g_new(jjj);
%                   end
%               end
               pp1=pop.cost;
               Cheb=max(abs(((pp1)'-repmat(Z,n_pop,1)).*W),[],2);
%                  aa=norm(W(i,:));
%                 g1=norm(abs((pop(i).cost-Z)/Z))/aa;
%                 g=norm(pop(i).cost-(Z+g1*W(i,:)/aa));
%                 Cheb(i)=g;
               [Cheb_rank,I]=sort(Cheb);
             elites_size=ceil(0.05*n_pop);
%              elites=pop(I(1:elites_size));
             elites=pop(1:elites_size);
             Cheb2=Cheb*(1-mul);
        end
        %Update p for each solution
%         Cheb=max(abs(((pp)'-repmat(Z,n_pop,1)).*W),[],2);
        deta1=abs((Cheb_old-Cheb));
%         deta=deta1/max(deta1)*0.5+0.3;
        deta=(deta1+0.1)/(max(deta1)+0.1)*0.3+0.2;

%         plot(deta,'*');
        % select the next generation
%          [pop, F, params] = select_pop(pop, params);
%          F1 = pop(F{1});
%          Fl = pop(F{end});
          F=nondominated_sort(pop);
          F1=pop(F{1});
          Fl = pop(F{end});
        % analysis
        iter_end = clock;
        avgfit = mean([F1.cost], 2);
        logger(['iter: ', num2str(iter), '/', num2str(n_iter), ' time: ', num2str(etime(iter_end, iter_start)), 's', ...
            ' fit: ', num2str(avgfit(1)), ', ', num2str(avgfit(2)), ', ', num2str(avgfit(3))]);
       % logger(['## accuracy = ', num2str(1 - best_member.cost(1)), ', features = ', num2str(round(n_feature * best_member.cost(2)))]);
%              if k==1
%             si=size(pop,1);
%              %si1=si1+si+1; 
%             for ii=1:si
%                  result(ii,1:n_feature)=[pop(ii).position*1];
%                  result(ii,n_feature+1:n_feature+3)=[pop(ii).cost];
%              end
%              xlswrite('Lymphoma(RAFS).xls', result(:,n_feature+1:n_feature+3),iter);      % 将result写入到wind.xls文件中
%              end
    end
    best_member = solution_selection(F1);
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
