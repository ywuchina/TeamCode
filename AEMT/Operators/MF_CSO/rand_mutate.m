function popm = rand_mutate(pop, n_feature, X, Y,mul_rate1,mul_rate2,F1,pp)
% mutation operation
    individual.position = [];
    individual.cost = [];
    popm = repmat(individual, 1, 1);
    popm.position=pop.position;

        position = quick_bit_mutate(popm.position,n_feature);
        popm.position = position;
      
       cost = fitness(X, Y,  popm.position);

%         popm.position = position;
        popm.cost = cost;
end

function new_position = quick_bit_mutate(position,n_feature)
    mu = 0.1;
    r = rand;

    index_one = find(position == true);
    index_zero = find(position == false);
    n_one = numel(index_one);
    n_zero = numel(index_zero);

    n_mu = ceil(min(n_one, n_zero) * mu);
    new_position = repmat(position, 1);
    index=randsample(n_feature, n_mu);
    new_position(index) = ~new_position(index);
%     if r > 0.5
%         index = randsample(index_one, n_mu);
%         new_position(index) = false;
%     else
%         index = randsample(index_zero, n_mu);
%         new_position(index) = true;
%     end
end