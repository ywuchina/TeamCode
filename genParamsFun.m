function params = genParamsFun(cloud_mov, cloud_ref, varargin)
% Set input parser
p = inputParser;
p.CaseSensitive = false;
p.addParameter('mode', 'point2plane');
p.addParameter('LieGroup', 1);
% p.addParameter('Mov_Normal', []);
% p.addParameter('Ref_Normal', []);
p.addParameter('minimizer', 'IRLS'); 
p.addParameter('Tf0', eye(4)); 
p.addParameter('P', [1.0 2.0]); 
p.addParameter('maxIter_icp', 50);
p.addParameter('maxIter_em', 10); % 20
p.addParameter('maxIter_irls', 10); % 5
% p.addParameter('maxInner_admm', 1);
% p.addParameter('maxOuter_admm', 10); 
p.addParameter('IS_SHOW', 0);
p.addParameter('verbose', 1);
parser = p;
parser.parse(varargin{:});
params = parser.Results; 
params.Mov0 = cloud_mov.Location'; 
params.Ref0 = cloud_ref.Location';
if ismember('Normal', fieldnames(cloud_mov))
    params.Mov_Normal = cloud_mov.Normal';
end
if ismember('Normal', fieldnames(cloud_ref))
    params.Ref_Normal = cloud_ref.Normal';
end
end