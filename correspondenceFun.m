function [params] = correspondenceFun(params, curTf)
rot = curTf(1:3, 1:3);
trt = curTf(1:3, end);
Aft = Loc2Glo(params.Mov0, rot', trt);
[NNIdx, DD] = knnsearch(params.MdRef, Aft');
EffIdx = find(DD <= inf);
%%%%%%% track the change.
params.Ref = params.Ref0(:, NNIdx(EffIdx));
params.Aft = Aft(:, EffIdx);
params.V   = params.Aft - params.Ref;
if strcmpi(params.mode, 'point2plane')
    params.Ref_Normal = params.Ref_Normal0(:, NNIdx(EffIdx));
    normal = params.Ref_Normal;
    if params.LieGroup
        n1 = normal(1, :);
        n2 = normal(2, :);
        n3 = normal(3, :);
        MArray = zeros(6, size(params.V, 2));
        MArray(1, :) = n1.^2;
        MArray(2, :) = n1.*n2;
        MArray(3, :) = n1.*n3;
        MArray(4, :) = n2.^2;
        MArray(5, :) = n2.*n3;
        MArray(6, :) = n3.^2;
        params.MArray = MArray;
    end
end
if strcmpi(params.mode, 'plane2plane')
    params.Omega_Ref = params.OmegaRef0(:, NNIdx(EffIdx));
    params.MArray    = cat(2, params.Omega_Ref(:).omega);
    params.LArray    = cat(2, params.Omega_Ref(:).L);
    params.invLArray = cat(2, params.Omega_Ref(:).invL);
end
params.Err = compute_residualFun(params);
end

