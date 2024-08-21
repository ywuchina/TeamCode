function Err = compute_residualFun(params)
Err = [];
V = params.V;
if strcmpi(params.mode, 'point2point')
    Err = sqrt(sum(V.^2));
end
if strcmpi(params.mode, 'point2plane')
    normal = params.Ref_Normal;
    tmp = V .* normal;
    Err = abs(sum(tmp));
end
if strcmpi(params.mode, 'plane2plane')
    MArray = params.MArray; 
    m1 = MArray(1, :);
    m2 = MArray(2, :);
    m3 = MArray(3, :);
    m4 = MArray(4, :);
    m5 = MArray(5, :);
    m6 = MArray(6, :);
    v1 = V(1, :); 
    v2 = V(2, :); 
    v3 = V(3, :); 
    Err = sqrt(m1.*v1.^2 + 2*m2.*v1.*v2 + 2*m3.*v1.*v3 + m4.*v2.^2 + 2*m5.*v2.*v3 + m6.*v3.^2); 
    scale = sqrt(1/1e-3); 
    Err = Err / scale; 
end
end
