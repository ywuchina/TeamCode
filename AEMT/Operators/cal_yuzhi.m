%计算第二临近平均距离
function md= cal_md(scan)  %scan（3XN）

Model= scan;
NS = createns(Model');  %1x1 KDTreeSearcher
[corr,TD] = knnsearch(NS,Model','k',2); %k为每一个点找俩最近点，corr是点索引，TD是距离  corr（NX2） TD（NX2）

% md=((2^0.5)/2)*(mean(TD(:,2))); % 二分之根号二 欧氏距离的均值----条件苛刻
md =sqrt(mean(TD(:,2).^2));       % L2的均值-----------------------平均分辨率
end    