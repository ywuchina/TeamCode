function [fitness_val,pop_position] = fitness2(X_all, Y, pop,prm)

    % no feature selected is not allowed
    n_feature=size(X_all,2);
    pop_position=pop.position();
    for j=1:n_feature
        if (pop.position(j)>0.5)&&(pop.taskfea(j)==1)
            position(j)=true;
        else
            position(j)=false;
            pop_position(j)=0;
        end
    end
    num_feature = sum(position);  % num of selected feature
    if num_feature == 0
        fitness_val = 1;
        return;
    end

    n_sample = size(X_all, 1);
    X = X_all(:, position);
    

    D = pdist(X);
    distances = squareform(D) / sqrt(num_feature);
    
    D2 = pdist(X, 'cityblock');
    distances2 = squareform(D2) / num_feature;
    
    

end

function class_indices = one_nn(train, test, distances)
    train_indices = find(train);
    test_indices = find(test);
    class_indices = zeros(size(test_indices, 1), 1);
    for k = 1:size(test_indices, 1) 
        i = test_indices(k); 
        dis_i = distances(i, train);
        [~, min_j] = min(dis_i);
        class_indices(k) = train_indices(min_j);
    end
end
