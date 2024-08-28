function fitness_val = fitness(X_all, Y, position)


    num_feature = sum(position);  % num of selected feature
    if num_feature == 0
        fitness_val = 1;
        return;
    end

    n_sample = size(X_all, 1);
    X = X_all(:, position);
    
    % Distance
    D = pdist(X);
    distances = squareform(D) / sqrt(num_feature);
    
    D2 = pdist(X, 'cityblock');
    distances2 = squareform(D2) / num_feature;
    
    % 1NN
    k = 10;
    indices = crossvalind('Kfold', Y, k);
    cp = classperf(Y);
    num_class = size(unique(Y, 'stable'), 1);
    for i = 1:k
        test = (indices == i); 
        train = ~test;  
        class_indices = one_nn(train, test, distances);
        class = Y(class_indices, :);
        classperf(cp, class, test);
    end
    error_distribution = cp.ErrorDistributionByClass;
    sample_distribution = cp.SampleDistributionByClass;
    err_distribution = error_distribution ./ sample_distribution;
    error_rate = sum(err_distribution) / num_class;


    feature_rate = num_feature / size(position, 2);

    db = zeros(1, n_sample);
    dw = zeros(1, n_sample);
    for i = 1:n_sample
        diff_indices = (Y ~= Y(i, :));
        same_indices = find(Y == Y(i, :));
        same_indices = same_indices(same_indices ~= i);
        db(1, i) = min(distances2(i, diff_indices));
        if isempty(same_indices)
            dw(1, i) = 0;
        else
            dw(1, i) = max(distances2(i, same_indices));  % may has no same_indices
        end
    end 
    
    db = sum(db) / n_sample;
    dw = sum(dw) / n_sample;
    distance = 1 / (1 + exp(-5 * (dw - db)));

   alfa=0.999999;
    fitness_val = alfa*alfa*error_rate+(1-alfa)*feature_rate+(1-alfa)*distance;

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
