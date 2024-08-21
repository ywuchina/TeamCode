
function [e_r, e_t, RMSE, MAE_R, MAE_t,RMSE_R, RMSE_t] = calc_errors(scan,R_pre,R_tru,t_pre,t_tru)
e_r = rad2deg(abs(acos((trace(R_pre'*R_tru)-1)/2)));       % 
e_t = norm(t_pre-t_tru);                                   % 

scan_data = scan.Location'; % 3XN 
scan_data_pretrans = transform_to_global(scan_data,R_pre,t_pre');
scan_data_trutrans = transform_to_global(scan_data,R_tru,t_tru');
% pcshowpair(pointCloud(double(scan_data_pretrans')),pointCloud(scan_data_trutrans'));
RMSE = sqrt(sum(sum((scan_data_pretrans-scan_data_trutrans).^2))/size(scan_data_pretrans,2)); % 

MAE_R = rad2deg(norm(rotationMatrixToVector(R_pre*R_tru'),1)/size(t_tru,1));              % 
MAE_t = norm(t_pre-t_tru,1)/size(t_tru,1);                                                % 

RMSE_R = rad2deg(sqrt((norm(rotationMatrixToVector(R_pre*R_tru'),2).^2)/size(t_tru,1)));  % 
RMSE_t = sqrt((norm(t_pre-t_tru,2).^2)/size(t_tru,1));                                    % 
 


end

