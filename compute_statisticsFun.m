function params = compute_statisticsFun(params0)
params = params0; 
params.MdRef = createns(params.Ref0');
if strcmpi(params.mode, 'point2plane')
    Ref_Normal = params.Ref_Normal; 
    params.Ref_Normal0 = Ref_Normal;
    if isempty(Ref_Normal)
         Normals = pcnormals(pointCloud(params.Ref0'), 10);
         params.Ref_Normal0 = Normals'; 
    end
end
if strcmpi(params.mode, 'plane2plane')
    kNum = 20;
    params.OmegaRef0 = computeCovFun(params.Ref0, kNum);
end
VBParams = [];
VBParams.tol     = 1e-5; % 1e-10;
VBParams.maxIter = params.maxIter_em;  % 
VBParams.IS_SHOW = 0;
VBParams.SPV   = params.P;
VBParams.alpha = 1e-3;
VBParams.a     = 1.0;
VBParams.b     = 1.0;
VBParams.reduceNum = 1e8; % params.ReduceNum; 
VBParams.EP_Plus = 1; 
params.VBParams = VBParams; 
%%%% compute resolution. 
% if strcmpi(params.mode, 'point2point')
%    [NNIdx, DD] = knnsearch(params.Ref0', params.Ref0', 'k', 2);
%    params.res = median(DD(:, 2)); 
% end
% if strcmpi(params.mode, 'point2plane')
%     
% end
end