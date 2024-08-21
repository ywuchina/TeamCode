function [dR, dT] = reg_point2planeFun(params)
if params.LieGroup
    [H, b] = CalHbFun(params, 1);
    dx = -(H + 0.0 * eye(size(H, 1)))\b;
    dR = expm(SkewFun(dx(1:3)));
    dT = dx(4:6);
else
    s = params.Aft;
    n = params.Ref_Normal;
    W = params.W;
    a = sum( n .* params.V)';
    Dim = size(s, 1);
    if Dim == 3
        a1 = n(3, :) .* s(2, :) - n(2, :) .* s(3, :);
        a2 = n(1, :) .* s(3, :) - n(3, :) .* s(1, :);
        a3 = n(2, :) .* s(1, :) - n(1, :) .* s(2, :);
        A = [a1' a2' a3' n'];
    else
        a1 = n(2, :).*s(1, :) - n(1, :).*s(2, :);
        A = [a1' n'];
    end
    if ~isrow(W)
        W = W';
    end
    W2 = sqrt(W);
    A = repmat(W2', 1, size(A, 2)) .* A;
    a = W2' .* a;
    %%%%%%%%%%% solve the problem || Ax - b ||, the solution also can be x = pinv( A' * A) * A' * b - x
    H = A'*A;
    b = A'*a;
    x = -pinv(H) * b;
    % [dR, dT] = ObtainTf(x);
    Ang = x(3:-1:1);
    dR = eul2rotm(Ang');
    dT = x(4:6);
end
end