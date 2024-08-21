
function [scan_data,R,t] = roteByRT(scan_model, Turbulence2R, Turbulence2T);


scan_sample = SampleForShape1(pointCloud(scan_model)); % 
MSEs= cal_mMSEs(scan_sample);    %
t = MSEs*(2*rand(3,1)-1)*Turbulence2T;      % Turbulence of t：    
theta= (2*pi*rand(3,1)-pi)*Turbulence2R;  % Turbulence of R      
M(1:3,1:3)= OulerToRota(theta);   % 3

M=rigid3d(M',theta');                              
tform = affine3d(M.T);
scan_data = pctransform(pointCloud(scan_model),tform);
scan_data=scan_data.Location;

M=inv(M.T);
M=M';

R=M(1:3,1:3);
t=M(1:3,4);

% pcshowpair(pointCloud(scan_model),pointCloud(scan_data))











end