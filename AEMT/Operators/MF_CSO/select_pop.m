function [new_pop, F, params] = select_pop(pop, params)

    F = nondominated_sort(pop);

    n_pop = params.n_pop;
    if numel(pop) == n_pop
        new_pop = pop;
        return;
    end

    new_pop = [];
    for l = 1:numel(F)
        if numel(new_pop) + numel(F{l}) > n_pop
            Fl = F{l};
            break;
        end
        new_pop = [new_pop; pop(F{l})];
    end

    St = [new_pop; pop(Fl)];
    St_Fl = numel(new_pop)+1:numel(St);

    [normalized_cost, params] = normalize(St, params);
    % pi是每个个体关联的参考点索引，d是个体与参考点的距离
    % 这里需要关联的种群应该是[F1, F2, ..., Fl]，即论文中的St
    [pi, d] = associate(normalized_cost, params.zr);
    nzr = size(d, 2);

    % rho是每个参考点在new_pop中的关联个数
    rho = zeros(1, nzr);
    for i = 1:numel(new_pop)
        rho(pi(i)) = rho(pi(i)) + 1;
    end

    while true
        [~, j] = min(rho);  % St中关联最少的引用点
        ref_Fl = []; % Fl中与j引用点关联的个体
        for i = St_Fl
            if pi(i) == j
                ref_Fl = [ref_Fl, i];
            end
        end

        if isempty(ref_Fl)
            rho(j) = inf;
            continue;
        end

        if rho(j) == 0
            ddj = d(ref_Fl, j);
            [~, new_member_index] = min(ddj);  % ref_Fl的索引
        else
            new_member_index = randi(numel(ref_Fl));  % ref_Fl的索引
        end

        new_member = ref_Fl(new_member_index);  % St中的索引
        new_pop = [new_pop; St(new_member)];

        rho(j) = rho(j) + 1;
        St_Fl(St_Fl == new_member) = [];  % 去掉添加的那个

        if numel(new_pop) >= n_pop
            break;
        end
    end

    F = nondominated_sort(new_pop);

end
