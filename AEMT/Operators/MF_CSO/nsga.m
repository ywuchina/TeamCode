function [best_member, F1] = nsga(X, Y,k)
    % parameter setting
    n_feature = size(X, 2);
    n_obj = 3;
    n_pop = min(round(n_feature / 20), 300);
    n_iter = 70;
    n_division = 27;
    n_mutation = n_pop;
    n_crossover = n_pop;

    % generate reference points
    zr = reference_points(n_obj, n_division);

    % inner parameters
    params.n_pop = n_pop;
    params.zr = zr;
    params.zmin = [];
    params.zmax = [];
    params.smin = [];

    % initialization
    individual.position = [];
    individual.cost = [];
    pop = repmat(individual, n_pop, 1);
    parfor i = 1:n_pop
        position = unifrnd(0, 1, 1, n_feature) > 0.5;
        pop(i).position = position;
        pop(i).cost = fitness(X, Y, position);
    end

    % init sort
    [pop, F, params] = select_pop(pop, params);
    F1 = pop(F{1});
    Fl = pop(F{end});

    % run iterations
    for iter = 1:n_iter
        iter_start = clock;
        % crossover
        popc = crossover_pop(pop, n_crossover, X, Y);
        combined_pop = [pop; popc];
        % mutation
        popm = mutate_pop(combined_pop, n_mutation, X, Y, Fl);
        combined_pop = [combined_pop; popm];
        % select the next generation
        [pop, F, params] = select_pop(combined_pop, params);
        F1 = pop(F{1});
        Fl = pop(F{end});

        % analysis
        iter_end = clock;
        avgfit = mean([F1.cost], 2);
        logger(['iter: ', num2str(iter), '/', num2str(n_iter), ' time: ', num2str(etime(iter_end, iter_start)), 's', ...
            ' fit: ', num2str(avgfit(1)), ', ', num2str(avgfit(2)), ', ', num2str(avgfit(3))]);
        % logger(['## accuracy = ', num2str(1 - best_member.cost(1)), ', features = ', num2str(round(n_feature * best_member.cost(2)))]);
%           if k==1
%           si=size(F1,1);
%           %si1=si1+si+1; 
%           for ii=1:si
%               result(ii,1:n_feature)=[F1(ii).position*1];
%               result(ii,n_feature+1:n_feature+3)=[F1(ii).cost];
%           end
%           xlswrite('Prostate6033(ps-nsga).xls', result(:,n_feature+1:n_feature+3),iter);      % 将result写入到wind.xls文件中
%           end
    end
    best_member = solution_selection(F1);
end

function best_member = solution_selection(F1)
    n_F1 = numel(F1);
    best_member = F1(1);
    for i = 2:n_F1
        if F1(i).cost(1) < best_member.cost(1)
            best_member = F1(i);
        elseif F1(i).cost(1) == best_member.cost(1) && F1(i).cost(2) < best_member.cost(2)
            best_member = F1(i);
        end
    end
end
