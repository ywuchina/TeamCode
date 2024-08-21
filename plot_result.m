clc; close all; clear all;  
DataRoot = 'output';
DataDir = fullfile(DataRoot, 'cloud_mov.pcd'); 
cloud_mov = pcread(DataDir); 
DataDir = fullfile(DataRoot, 'cloud_ref.pcd'); 
cloud_ref = pcread(DataDir); 
figure; 
hold on; 
grid on; 
pcshow(cloud_mov); 
pcshow(cloud_ref.Location, 'b'); 
DataDir = fullfile(DataRoot, 'keypoints_mov.pcd'); 
cloud_key_points_mov = pcread(DataDir); 
pcshow(cloud_key_points_mov.Location, 'r', 'markersize', 100); 
DataDir = fullfile(DataRoot, 'keypoints_ref.pcd'); 
cloud_key_points_ref = pcread(DataDir); 
pcshow(cloud_key_points_ref.Location, 'k', 'markersize', 100); 
%% 
DataDir = fullfile(DataRoot, 'coarse_results.txt'); 
A = importdata(DataDir);
str = A.textdata;
str = strtrim(str);
Headers = split(str);
[~, pos_t] = ismember({'tx', 'ty', 'tz'}, Headers);
[~, pos_q] = ismember({'qw', 'qx', 'qy', 'qz'}, Headers);

data = A.data; 
dT = data(pos_t)'; 
quat = data(pos_q); 
dR = quat2rotm(quat);
aft = Loc2Glo(cloud_mov.Location', dR', dT); 

figure; 
hold on; 
grid on; 
axis equal; 
xlabel('X/m'); 
ylabel('Y/m'); 
pcshow(cloud_ref.Location, 'g', 'markersize', 100); 
pcshow(cloud_mov.Location, 'r', 'markersize', 100); 
pcshow(aft', 'b', 'markersize', 100); 
legend({'Ref', 'Mov', 'Coarse Aligned'}, 'FontSize', 12, 'Location', 'best');
legend('boxoff'); 

