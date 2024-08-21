function [InfoArray] = computeCovFun(data, kNum)
InfoArray = [];
[NNIdx, ~] = knnsearch(data', data', 'k', kNum);
for id = 1 : 1 : length(NNIdx)
    idx = NNIdx(id, :);
    C = cov(data(:, idx)');
    [U, S, ~] = svd(C);
    SS = diag(S);
    epsilon = 0.001;
    tmp = [];
    Cov = U * diag([1.0, 1.0, epsilon]) * U';
    Omega = U * diag([1.0, 1.0, 1.0/epsilon]) * U';
    L = chol(Omega);
    invL = inv(L);
    tmp.cov   = vectorizeFun(Cov); 
    tmp.omega = vectorizeFun(Omega);
    tmp.L     = vectorizeFun(L);
    tmp.invL  = vectorizeFun(invL);
    InfoArray = [InfoArray tmp];
end
end
function a = vectorizeFun(A)
    a = zeros(6, 1); 
    a(1) = A(1, 1); 
    a(2) = A(1, 2); 
    a(3) = A(1, 3); 
    a(4) = A(2, 2); 
    a(5) = A(2, 3); 
    a(6) = A(3, 3); 
end

