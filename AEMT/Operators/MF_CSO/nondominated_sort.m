function F = nondominated_sort(pop)
% non-dominated sorting
    num_pop = numel(pop);
    for i = 1:num_pop
        domination_set{i} = [];
        dominated_cnt{i} = 0;
    end

    F{1} = [];
    for i = 1:num_pop
        for j = i+1:num_pop
            if dominates(pop(i).cost, pop(j).cost)
                domination_set{i} = [domination_set{i}, j];
                dominated_cnt{j} = dominated_cnt{j} + 1;
            end
            if dominates(pop(j).cost, pop(i).cost)
                domination_set{j} = [domination_set{j}, i];
                dominated_cnt{i} = dominated_cnt{i} + 1;
            end
        end
        if dominated_cnt{i} == 0
            F{1} = [F{1}, i];
        end
    end

    k = 1;
    while true
        Q = []; % Q is used to store F in k+1 layer
        for i = F{k}
            for j = domination_set{i}
                dominated_cnt{j} = dominated_cnt{j} - 1;
                if dominated_cnt{j} == 0
                    Q = [Q, j];
                end
            end
        end
        if isempty(Q)
            break;
        end
        F{k + 1} = Q;
        k = k + 1;
    end

end
