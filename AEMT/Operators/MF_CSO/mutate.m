function popm = mutate(pop, n_feature, X, Y,mul_rate1,mul_rate2,F1,pp)
% mutation operation
    individual.position = [];
    individual.cost = [];
    popm = repmat(individual, 1, 1);
    popm.position=pop.position;
      for i=1:n_feature
          tt=rand;
%            if mul_rate1(i)<tt
          if pp>tt
              popm.position(i)=0;
          end
      end
      %repair
%         p=randperm(size(F1,1));
%         elites=F1(p(1));
%          for i=1:n_feature
%             ttt=rand;
%             if mul_rate2(i)>ttt
%                 popm.position(i)=elites.position(i);
%             end
%         end
        
      
      
       cost = fitness(X, Y,  popm.position);

%         popm.position = position;
        popm.cost = cost;
end

