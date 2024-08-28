function popm = mutate_pop(pop, n_mutation, X, Y,Fl)
% mutation operation
    individual.position = [];
    individual.cost = [];
    popm = repmat(individual, n_mutation, 1);
    parfor k = 1:n_mutation

         max_retry = 3;
         for t = 1:max_retry
 
             p = pop(randi(numel(pop)));
             position = quick_bit_mutate(p.position);
 
             cost = fitness(X, Y, position);
             if ~is_dominated(cost, Fl)
                 break;
             end
 
         end
        
%         % no mutation retry
%          p = pop(randi(numel(pop)));
%          position = quick_bit_mutate(p.position);
%          cost = fitness(X, Y, position);
% 
         popm(k).position = position;
         popm(k).cost = cost;
%          if mod(k, 100) == 0
%             logger(['mutate ', num2str(k), '/', num2str(n_mutation), ', cost = ', num2str(pop(k).cost')]);
%         end
    end
end

function new_position = quick_bit_mutate(position)
    mu = 0.1;
    r = rand;

    index_one = find(position == true);
    index_zero = find(position == false);
    n_one = numel(index_one);
    n_zero = numel(index_zero);

    n_mu = ceil(min(n_one, n_zero) * mu);
    new_position = repmat(position, 1);
    if r > 0.5
        index = randsample(index_one, n_mu);
        new_position(index) = false;
    else
        index = randsample(index_zero, n_mu);
        new_position(index) = true;
    end
end

function b = is_dominated(cost, Fl)
    b = false;
    for i = 1:numel(Fl)
        if dominates(Fl(i).cost, cost)
            b = true;
            break;
        end
    end
end

% function b = inpop(position, pop)
%     b = false;
%     for i = 1:numel(pop)
%         if all(pop(i).position == position)
%             b = true;
%             break;
%         end
%     end
% end
