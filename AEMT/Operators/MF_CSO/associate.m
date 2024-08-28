function [pi, d] = associate(normalized_cost, zr)
% 关联参考点
    nzr = size(zr, 2);
    num_pop = size(normalized_cost, 2);

    pi = zeros(num_pop, 1);
    d = zeros(num_pop, nzr);

    for i = 1:num_pop
        for j = 1:nzr
            w = zr(:, j) / norm(zr(:, j));
            z = normalized_cost(:, i);
            d(i, j) = norm(z - w' * z * w);
        end

        [~, jmin] = min(d(i, :));
        pi(i) = jmin;
    end

end
