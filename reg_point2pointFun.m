%%%%%%%%%%%%%%% Calculate the registration matrix %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% T(TData)->MData %%%%%%%%%%%%%%%%%%%%%%%%%
% SVD solution
function [R1, t1] = reg_point2pointFun(varargin)
if nargin == 1
    params = varargin{1}; 
end
if nargin == 2
    params = [];
    params.Aft = varargin{1};
    params.Ref = varargin{2};
    params.W   = ones(1, size(params.Ref, 2));
end
if nargin == 3
    params = []; 
    params.Aft = varargin{1}; 
    params.Ref = varargin{2}; 
    params.W   = varargin{3}; 
end
MovData = params.Aft;
RefData = params.Ref;
W       = params.W;
%%%%%%% normalize the weight.
Dim = size(MovData, 1);
W = W / sum(W);
W_Normalize = repmat(W, Dim, 1);
M = RefData;
mm = sum(M.*W_Normalize, 2);
S = MovData;
ms = sum(S.*W_Normalize, 2);
Sshifted = bsxfun(@minus, S, ms );
Mshifted = bsxfun(@minus, M, mm );

Sshifted = Sshifted .* W_Normalize;
K = Sshifted*Mshifted';
[U A V] = svd(K);
R1 = V*U';
if det(R1)<0
    B = eye(Dim);
    B(Dim,Dim) = det(V*U');
    R1 = V*B*U';
end
t1 = mm - R1*ms;
end