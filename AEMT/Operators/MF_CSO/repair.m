function offspring = repair(popm,elite,n_feature,gel,gof,X,Y,p1,p2)
% mutation operation
    individual.position = [];
    individual.cost = [];
    offspring = repmat(individual, 1, 1);
    offspring.position=popm.position;
   
      %repair
%       p=randperm(size(F1,1));
%       elites=F1(p(1));
%        for i=1:n_feature
%           ttt=rand;
%           if mul_rate2(i)>ttt
%               popm.position(i)=elites.position(i);
%           end
%       end
%    
%        if gel>gof
%          offspring=popm;
%        else
%           p=abs(gof-gel)/gel;
        for i=1:n_feature
            if gel(i)>gof(i)
                offspring.position(i)=popm.position(i);
            else
               if ((p2(i)>rand)&&(elite.position(i)==1))
                  offspring.position(i)=elite.position(i);
               end 
            end
        end
       
%       end
      
      
       cost = fitness(X, Y,  offspring.position);

%         popm.position = position;
        offspring.cost = cost;
end
