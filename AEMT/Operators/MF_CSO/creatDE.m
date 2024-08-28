function popc=creatDE(ppop,n_pop,X,Y,iter)
%%pop里存放的是第一层上的点
M=2;
CR=0.4;%%CR取值在[0,1];
U=[];
s1=size(ppop,1);
F=0.5;
%变异
individual.position = [];
individual.cost = [];
popc = repmat(individual, 1, 1);
SET.position = [];
SET.cost = [];
SET2 = repmat(SET, 2, 1);
i=1;
% for i=1:s1
%     p=randperm(s2);%%p中存储的是随机打乱顺序的1,2,3,4,5...序列
%     pp=randperm(s1);
%     h=ceil(rand*n_pop);
%   if s2>3
%        SET1=F1([p(1:3)],:); 
%   else
%           addr=find(pp==i);%找到i在p中的位置
%      if (addr+3)<=s1 %%保证不会超出max_sizeP，同时随机选取三个向量
%          SET1=ppop([pp(addr+1:addr+3)],:);
%      else
%                   SET1=ppop([pp(addr-3:addr-1)],:);
%     end
%   end
SET1=ppop(2:4);
      
%选出best
         if (is_dominated(SET1(1).cost, SET1(2)))&&(is_dominated(SET1(1).cost, SET1(3)))
                   ran1=1;
         elseif (is_dominated(SET1(2).cost, SET1(1)))&&(is_dominated(SET1(2).cost, SET1(3)))
                   ran1=2;
         else
             ran1=3;
         end
%      ran1=randperm(3,1);
     BAS_V=(SET1(ran1));
 %除去最优向量后的另外两个向量
     jj=1;
    for ii=1:3
        if ii~=ran1
           SET2(jj)=SET1(ii);
           jj=jj+1;
        end
    end  
%modified
         % P=BAS_V.position(1:n_pop)+F*abs(SET2(1).position(1:n_pop)-SET2(2).position(1:n_pop));
%           P=BAS_V.position(1:n_pop)+F*abs(SET2(1).position(1:n_pop)-SET2(2).position(1:n_pop));
%          rand1=rand;
%          P(P>rand1)=1;
%          P(P==rand1)=0;
%          P(P<rand1)=0;
%              for j=1:n_pop
%                   BAS_V.position(j)=P(j);
%              end
%origin
          P=abs(F*rand*(SET2(1).position(1:n_pop)-SET2(2).position(1:n_pop)))+0.01;
          for j=1:n_pop
              if P(i,j)>rand
                   BAS_V.position(j)=1-BAS_V.position(j);
              end
          end

        BAS_V.cost = fitness(X, Y, BAS_V.position);
    U1=crossProb(ppop(i),BAS_V,CR,n_pop,X,Y,iter);
    popc(i).position=U1.position;
    popc(i).cost=U1.cost;
% end


function U=crossProb(par,V,CR,k,X,Y,n_iter)%%交叉
h=ceil(rand*k);
for j=1:k
    uu=rand;
    if ((uu<CR)|(j==h))
        U(1).position(j)=V(1).position(j);
    else
        U(1).position(j)=par(1).position(j);
    end
end
 if sum(U(1).position(1:k))==0
     aa=randperm(k);
     for i=1:k
         if rand<0.5
              U(1).position(aa(i))=1;
             break;
         end
     end 
 end
 U(1).cost = fitness(X, Y, U(1).position);

function b = is_dominated(cost, Fl)
    b = false;
    for i = 1:numel(Fl)
        if dominates(Fl(i).cost, cost)
            b = true;
            break;
        end
    end
