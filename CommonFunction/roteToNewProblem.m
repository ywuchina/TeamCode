
function [data, GrtR, GrtT] = roteToNewProblem(data, GrtR, GrtT, Turbulence2R, Turbulence2T, sample_ration)


scannum= size(data,2);                                   
for i=2:scannum                                           
    GrtM = [GrtR{1,i},GrtT{1,i};0 0 0 1];
    tform = affine3d(GrtM');
    scan_data = pctransform(pointCloud(double(data{1,i}')),tform);
    data{1,i} = scan_data.Location'; 
    GrtR{1,i} = eye(3);
    GrtT{1,i} = [0 0 0]';
    MSEs = Turbulence2T;
    t = MSEs*(2*rand(3,1)-1)*Turbulence2T;      %     
    theta= (2*pi*rand(3,1)-pi)*Turbulence2R;  % 
    M(1:3,1:3)= OulerToRota(theta);   % 
    M=rigid3dDHQ(M',t);                              
    tform = affine3d(M);
    scan_data = pctransform(pointCloud(double(data{1,i}')),tform);
    %% 
    data{1,i} = scan_data.Location';
    M=inv(M);
    M=M';
    %% 
    GrtR{1,i} = M(1:3,1:3);
    GrtT{1,i} = M(1:3,4);  
end

end