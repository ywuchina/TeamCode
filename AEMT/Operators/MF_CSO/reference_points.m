function zr = reference_points(n_obj, n_division)
% generate reference points
    zr = get_fixed_row_sum_int_matrix(n_obj, n_division)' / n_division;
end

function A = get_fixed_row_sum_int_matrix(m, row_sum)

    if m == 1
        A = row_sum;
        return;
    end

    A = [];
    for i = 0:row_sum
        B = get_fixed_row_sum_int_matrix(m - 1, row_sum - i);
        A = [A; i * ones(size(B, 1), 1), B];
    end

end
