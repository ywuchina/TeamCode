% main
clc;
clear;
close all;
warning('off');
diary 'log.txt';
% parpool(24); % num of parallel executor

result_file = 'result_members.txt';

% read data
 %load('data2/warpPIE10P.mat');
  load('data/srbct.mat');
   X = data(:, 2:end);
   Y = data(:, 1);
num_feature = size(X, 2);
% logger(['num of feature = ', num2str(num_feature)]);
start_time = clock;

% normalization
X_norm = (X - repmat(min(X), size(X, 1), 1)) ./ repmat(max(X) - min(X), size(X, 1), 1);

% t_max rounds
t_max = 1;
result_members_all = [];
cost_F1_all = [];
% hvs = []; spacings = [];
for t = 1:t_max

    % k fold cross-validation
    k = 10;
    indices = crossvalind('Kfold', Y, k);
    member.train_acc = 0; member.test_acc = 0; member.n_feature = 0;
    result_members = repmat(member, k, 1);
    for i = 1:k
        fold_start = clock;

        test = (indices == i);  % indices of i fold
        train = ~test;
        % split training set and testing set     
        X_train = X_norm(train, :);
        X_test = X_norm(test, :);
        Y_train = Y(train, :);
        Y_test = Y(test, :);

        % run nsga and get a solution
%         [best_member, F1] = nsga(X_train, Y_train,i);
%         [best_member, F1] = MOEAD(X_train, Y_train,i);
        [best_member, F1] = multitask(X_train, Y_train,i);
        X_train_best = X_train(:, best_member.position);
        X_test_best = X_test(:, best_member.position);

        % train and predict
        Mdl = fitcknn(X_train_best, Y_train, 'NumNeighbors', 1);
        class = predict(Mdl, X_test_best);

        % calculate accuracy
        cp = classperf(Y_test);
        num_class = size(unique(Y_test, 'stable'), 1);
        classperf(cp, class);
        error_distribution = cp.ErrorDistributionByClass;
        sample_distribution = cp.SampleDistributionByClass;
        acc_distribution = (sample_distribution - error_distribution) ./ sample_distribution;
        accuracy = sum(acc_distribution) / num_class;

        % save the solution
        result_members(i).train_acc = 1 - best_member.cost(1);
        result_members(i).test_acc = accuracy;
        result_members(i).n_feature = sum(best_member.position);
        
        % analyze F1
        num_F1 = size(F1, 1);
        costs = [F1.cost];
        positions = [F1.position];
        positions = reshape(positions, num_feature, num_F1)';
        test_accs = zeros(1, num_F1);
        for j = 1:num_F1
            X_train_F1 = X_train(:, positions(j, :));
            X_test_F1 = X_test(:, positions(j, :));
            Mdl = fitcknn(X_train_F1, Y_train, 'NumNeighbors', 1);
            class = predict(Mdl, X_test_F1);
            cp = classperf(Y_test);
            num_class = size(unique(Y_test, 'stable'), 1);
            classperf(cp, class);
            error_distribution = cp.ErrorDistributionByClass;
            sample_distribution = cp.SampleDistributionByClass;
            acc_distribution = (sample_distribution - error_distribution) ./ sample_distribution;
            accuracy = sum(acc_distribution) / num_class;
            test_accs(1, j) = accuracy;
        end
        costs = [costs; test_accs];
        cost_F1_all = [cost_F1_all costs];

        % analysis
%         PopObj = [F1.cost]';
%         hv = HV(PopObj, [0, 0, 0]);
%         spacing = Spacing(PopObj);
%         hvs = [hvs; hv]; spacings = [spacings; spacing];
        fold_end = clock;
        prefix = strcat('round', num2str(t), '/', num2str(t_max), ' fold', num2str(i), '/', num2str(k), ' - ');
        % logger([prefix, ' time:', num2str(etime(fold_end, fold_start)), 's',...
        %     ' train_acc:', num2str(result_members(i).train_acc), ...
        %     ' test_acc:', num2str(result_members(i).test_acc), ...
        %     ' size:', num2str(result_members(i).n_feature)]);
%         logger(['## spacing = ', num2str(spacing), ', hv = ', num2str(hv)]);
    end
    % save result every round
    dlmwrite(result_file, cell2mat(struct2cell(result_members))', '-append');
    result_members_all = [result_members_all; result_members];
end

% analysis
end_time = clock;
total_time = etime(end_time, start_time);
logger(['total time consumed = ', num2str(total_time), 's / ', num2str(total_time/60), 'min']);
mean_train_acc = mean([result_members_all.train_acc]);
mean_test_acc = mean([result_members_all.test_acc]);
mean_feature_size = mean([result_members_all.n_feature]);
std_test_acc = std([result_members_all.test_acc] * 100);
% dlmwrite(result_file, cell2mat(struct2cell(result_members_all))', '-append')
logger(['average training accuracy = ', num2str(mean_train_acc)]);
logger(['average testing accuracy = ', num2str(mean_test_acc)]);
logger(['std testing accuracy = ', num2str(std_test_acc)]);
logger(['average feature size = ', num2str(mean_feature_size)]);

dlmwrite('round_cost.out', cost_F1_all', '-append');

diary off;
