function [e_r_RSICP, e_t_RSICP,RMSE,time,T0] = call_RSICP(R_real,t_real,src,tar) 

Tini_gt = [R_real t_real;0 0 0 1];
source = src;     
target = tar;     

SP = double(source.Location');   
SN = pcnormals(src,3)';      
TP = double(target.Location');    
TN = pcnormals(tar,3)';      

scaleS = norm(max(SP,[],2)-min(SP,[],2)); % 2.08
scaleT = norm(max(TP,[],2)-min(TP,[],2)); % 2.14
scale = max(scaleS,scaleT);
SP = SP/scale;
TP = TP/scale;

meanS = mean(SP,2);
meanT = mean(TP,2);
SP = SP-repmat(meanS,1,size(SP,2));
TP = TP-repmat(meanT,1,size(TP,2));


t1 = clock;
[T0, count] = RSICP(SP,TP,SN,TN);
t2 = clock;
time = etime(t2,t1);

trans = T0(1:3,4);
trans = trans + meanT - T0(1:3,1:3) * meanS;
trans = trans*scale;
T0(1:3,4)=trans;

SP = double(source.Location');
P1 = T0(1:3,1:3)*SP+repmat(T0(1:3,4),1,size(SP,2));
P2 = Tini_gt(1:3,1:3)*SP+repmat(Tini_gt(1:3,4),1,size(SP,2)); 
% pcshowpair(pointCloud(P1'),pointCloud(P2'))
% rmse = sqrt(sum(sum((P1-P2).^2))/size(SP,2))

[e_r_RSICP,e_t_RSICP] = calc_error(T0(1:3,1:3),R_real,T0(1:3,4),t_real);  
RMSE = cal_error_rmse(src,T0(1:3,1:3),R_real,T0(1:3,4),t_real); 


end