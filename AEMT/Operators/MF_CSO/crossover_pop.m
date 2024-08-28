function popc = crossover_pop(pop, n_crossover, X, Y)
% crossover
    individual.position = [];
    individual.cost = [];
    popc = repmat(individual, n_crossover, 1);
    
    parfor k = 1:n_crossover
        
        parents = randsample(pop, 2, 1);
        r = unifrnd(0, 1);
%         if r > 0.1
            position = simple_crossover(parents(1).position, parents(2).position);
%         else
%             position = combine_crossover(parents(1).position, parents(2).position);
%         end
        popc(k).position = position;
        popc(k).cost = fitness(X, Y, position);

    end
end

function new_position = simple_crossover(position1, position2)
    new_position = repmat(position1, 1);
    index = randsample(numel(position2), round(numel(position2) / 2));
    new_position(index) = position2(index);
end

function new_position = combine_crossover(position1, position2)
    new_position = repmat(position1, 1);
    new_position(position2) = true;
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
