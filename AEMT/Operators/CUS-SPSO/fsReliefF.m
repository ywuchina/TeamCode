function [out] = fsReliefF(X, Y, k, m)
% function [wFeat] = fsReliefF(X, Y, k, m)
%   X,  the features on current trunk, each colum is a feature vector on all
%   instances, and each row is a part of the instance
%   Y,  the label of instances, in single column form: 1 2 3 4 5 ...
%   param consists of SimW, k and m
%       k,  the size of the neighborhood
%       m,  how many samples we want to try
%   out.W,  the weight of the features.

nF = size(X,2);

if nargin <= 2
    k = 10;
end
if nargin <= 3
    m = -1;
end

config = wekaArgumentString({'-M', m, '-D', 1 '-K', k});
t = weka.attributeSelection.ReliefFAttributeEval();
t.setOptions(config);
t.buildEvaluator(wekaCategoricalData(X,SY2MY(Y)));

out.W = zeros(1,nF);

for i =1:nF
    out.W(i) = t.evaluateAttribute(i-1);
end

%[throwAway, out.fList] = sort(out.W, 'descend');
out.fList = sort(out.W, 'descend');
out.prf = -1;
end