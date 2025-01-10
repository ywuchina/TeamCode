%% 以欧氏距离为度量标准，进行最远点采样，返回索引
function [ S ] = fps_euclidean(V, n, seed)
%EUCLIDEANFPS Samples K vertices from V by using farthest point sampling.
% The farthest point sampling starts with vertex v1 and uses the euclidean
% metric of the 3-dimensional embedding space.
% -  V is a n-by-3 matrix storing the positions of n vertices
% -  K(g:n) is the number of samples
% -  v1(g:seed) is the index of the first vertex in the sample set. (1<=v1<=n)
% Returns
% -  S is a K-dimensional vector that includes the indeces of the K sample
%    vertices.

%Hint: matlab function pdist2 could be helpful

S = zeros(n,1);
S(1) = seed;
d = pdist2(V,V(seed,:));    % 计算V中每个点到V(seed)的距离

% 两个集合：查询点集S、剩余点集（代码中没有标出）
% 剩余点a到点集S的距离：a到S中所有点距离的最小值
% 取到点集S距离最大的剩余点a_max加入点集S

for i=2:n
    [~,m] = max(d);         
    S(i) = m(1);                        % 将剩余点a加入点集S
    d = min(pdist2(V, V(S(i),:) ), d);  % 后一个d是a加入前距离向量，代表了剩余点到点集S的距离
                % 加入前距离向量中最大元素的索引是m(1)，pdist2(V, V(S(i) )会使d(m(1))变为0
                % 因为V(m(1))以加入点集S，V(m(1))到点集S的距离是0
end

end
