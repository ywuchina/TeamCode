function [r_overlap, Mrmse]= overlap(src,tar, T,yuzhi)

% threshold = sqrt(2)/2*threshold;
yuzhi = yuzhi;               % target 的 resolution

%% 根据 T 得到新的src
src=pctransform(src,affine3d(T));
src=src.Location;
num=size(src, 1);   
% src = src.Location;  % NX3   
% data_temp=[src,ones(num,1)]; % NX4
% data_temp=T*data_temp';  % 4XN 
% src=data_temp(1:3,:)';       % NX3   % 变换准确无误  src=pointCloud(src);tar=pointCloud(tar);pcshowpair(src,tar)

tar =tar.Location;
kdtree = KDTreeSearcher(tar);
[idx, dist] = knnsearch(kdtree,src);

Mrmse = sqrt(mean(dist.^2));

r_overlap = sum(dist <= yuzhi) / num;
    
end

