function [outParams] = MoEFun(params)
%% we need warm up. 
outParams = [];
params = compute_statisticsFun(params);
nCnt = 0;
ELBO = -inf;
tol = 1e-4; % 1e-4;
TModel = [];
MedianArray = [];
dTNormArray = [];
Tf0 = params.Tf0;
if params.IS_SHOW
    cloud_mov = pointCloud( params.Mov0'); 
end
for nIter = 1 : 1 : params.maxIter_icp
    curTf = Tf0;
    if ~isempty(TModel)
        curTf = TModel(end).Tf;
    end
    %%%%%% correspondence finder.
    params = correspondenceFun(params, curTf);
    med_val = quantile(params.Err, 0.5);
    %%%%%%%%% variational inference.
    params.VBParams.maxIter = params.maxIter_em; % min( round(vb_iter), params.maxIter_em );
    [model, ~, Z, L]  = VBMoEFun(params.Err, params.VBParams, 0);
    %%%%%%%
    [dR, dT] = IRLSFun( params, Z, model, params.maxIter_irls );
    dTf = [dR dT; 0 0 0 1];
    MedianArray(end+1) = median(params.Err); % length(find(params.Err <= 0.1));
    lastTf = Tf0;
    if ~isempty(TModel)
        lastTf = TModel(end).Tf;
    end
    Tf_new = dTf * lastTf;
    %%%% terminate condition.
    ELBO(end+1) = L(end);
    %%%%%%% make sure that the iteration is larger engough for convergence.
    dTNormArray(end+1) = norm(dT);
    dT = dTf(1:3, end);
    if norm(dT) <= tol
        nCnt = nCnt + 1;
    end
%     if nIter >= 2
%         Diff = abs(MedianArray(end) - MedianArray(end-1));
%         if Diff <= tol
%             % nCnt = nCnt + 1;
%         end
%     end
    if nCnt >= 3
        break;
    end
    %%%% record information.
    tmp = [];
    tmp.model = model;
    tmp.Tf = Tf_new;
    tmp.llh = ELBO(end);
    TModel = [TModel tmp];
    params.Z = Z;
    str = sprintf('nIter = %03d, mode = %s, K = %02d, med_val = %.2f, elbo = %+06.6f', ...
        nIter, params.mode, length(model), med_val, ELBO(end) );
    if params.verbose
        disp(str);
    end
end
if params.IS_SHOW
    params.VBParams.maxIter = 100; % min( round(vb_iter), params.maxIter_em );
    params.save = 1;
    [model, ~, ~, ~]  = VBMoEFun(params.Err, params.VBParams, 0);
    if length(model) == 2
        VisualizeFun(model, params);
        str = sprintf('nIter = %02d', nIter);
        title(str);
    end
end
rstTf = TModel(end).Tf;
outParams.rstTf  = rstTf;
outParams.ELBO   = ELBO;
outParams.TModel = TModel;
outParams.MedianArray  = MedianArray;
outParams.dTNormArray  = dTNormArray;
if params.IS_SHOW
    rstR = rstTf(1:3, 1:3);
    rstT = rstTf(1:3, end);
    Aft = Loc2Glo( params.Mov0, rstR', rstT );
    figure;
    hold on;
    grid on;
    axis equal;
    view(3);
    pcshow(params.Ref0', 'g', 'markersize', 50);
    pcshow(Aft', 'b',  'markersize', 50);
    Ang = rad2deg(rotm2eul(rstR));
    legend({'Target', 'Transformed Source'}, 'FontSize', 16, 'FontWeight', 'bold', ...
        'box', 'off', 'location', 'best');
    str = sprintf('RegResult of MoE-ICP, dT = %.3fm, dR = %.3fdeg', norm(rstT), Ang(1));
    title(str);
    figure;
    hold on;
    grid on;
    plot(ELBO, 'b.-');
    title('ELBO Curve');
    figure;
    hold on;
    grid on;
    plot(-MedianArray, 'b.-');
    plot(-dTNormArray, 'r.-');
    legend({'Median', 'dTNorm'}, 'FontSize', 14, 'box', 'off', 'location', 'best');
    title('Incremental Curves');
end