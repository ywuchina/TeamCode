function [zmax, smin] = scalarize(fp, zmax, smin)

    n_obj = size(fp, 1);

    if isempty(smin)
        zmax = inf(n_obj, n_obj);
        smin = inf(1, n_obj);
    end

    for j = 1:n_obj

        w = get_scalarizing_vector(n_obj, j);
        s = max(fp ./ repmat(w, 1, size(fp, 2)));  % 每一列的最大值的行向量

        [sminj, ind] = min(s);

        if sminj < smin(j) && is_valid(fp(:, ind), zmax, j)
            zmax(:, j) = fp(:, ind);
            smin(j) = sminj;
        end

    end

end

function w = get_scalarizing_vector(n_obj, j)
    epsilon = 1e-10;
    w = epsilon * ones(n_obj, 1);
    w(j) = 1;
end

function b = is_valid(cost, zmax, j)
    b = true;
    zmax(:, j) = cost;
    if det(zmax) == 0
        b = false;
    end
end
