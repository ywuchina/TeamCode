
function md= cal_md(scan)  %scan（3XN）

Model= scan;
NS = createns(Model');  %1x1 KDTreeSearcher
[corr,TD] = knnsearch(NS,Model','k',2); %k

% md=((2^0.5)/2)*(mean(TD(:,2))); % 
md =sqrt(mean(TD(:,2).^2));       % 
    