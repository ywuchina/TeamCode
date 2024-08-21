
tic;clc;clear all;close all; 
addpath(genpath(pwd));
load happy_stand_Identity

target=1;                              
source=2;                            
param.mu= 0.02 ;                      
param.sigma=0.6 ;                      
pop=100;                                         
gen=60;                                 
pop_S=pop/1;                                   
selection_pressure = 'roulette wheel'; 
p_il = 0;                                   
rmp=0.5;                               
scannum= size(data,2);                 
gridsize = 0.009;                     
sample_ration=0.1;                     
rot_m = eye(3);                                                         
for i = 1:scannum  
    data{1,i} = pcdenoise(pointCloud(data{1,i}'));                       
    sample_pc{i} = pcdownsample(data{1,i},'gridAverage',gridsize);     
    [norm_pcs{i}, norm_trans{i}]= Loca_normalization(sample_pc{i});      
    M=rigid3d(rot_m',-norm_trans{i});                                    
    tform = affine3d(M.T);
    data{1,i} = pctransform(data{1,i},tform);                            
    GrtT{i}=GrtT{i}+GrtR{i}*norm_trans{i}';
end 
for i = 2:scannum                                                                                                      
    GrtT{i} = inv(GrtR{1})*(GrtT{i}-GrtT{1});
    GrtR{i} = inv(GrtR{1})*GrtR{i};
end
GrtT{1}=[0 0 0]';GrtR{1}=eye(3);                                         
pc{1} = data{1, target};           
pc{2} = data{1, source};           
norm_pc{1} = norm_pcs{1, target};  
norm_pc{2} = norm_pcs{1, source};  
R_real=GrtR{1, source};            
t_real=GrtT{1, source};             
%M=rigid3d(R_real', t_real');  tform = affine3d(M.T);t_pc1 = pctransform(pc{2}, tform);t_pc2 = pc{1};pcshowpair(t_pc1,t_pc2)
disp('++++++++   Start the point cloud registration method   ++++++++');
clear Tasks
rotation=2;                                                               
n = 6;                                                                     
l_trans = 0.5*min([norm_pc{1}.Location; norm_pc{2}.Location]);
u_trans = 0.5*max([norm_pc{1}.Location; norm_pc{2}.Location]);
l_rot = (-pi)/rotation * ones(1,3);                                        
u_rot = pi/rotation * ones(1,3);
param.iter_huber = 3*max(u_trans-l_trans);                                   
param.iter_huber_end = 2*cal_md(norm_pc{1}.Location');                       
param.anneal_rate = 10.^(log10(param.iter_huber_end/param.iter_huber)/(gen)); 
iter = log10(param.iter_huber_end/param.iter_huber)/log10(param.anneal_rate);
            
Tasks(1).dims = n;
Tasks(1).pc1 = pc{2};   
Tasks(1).pc2 = pc{1};
Tasks(1).norm_pc1 = norm_pc{2};
Tasks(1).norm_pc2 = norm_pc{1};
Tasks(1).resolution = cal_md(norm_pc{1}.Location');   
Tasks(1).kdtree = KDTreeSearcher(Tasks(1).norm_pc2.Location);
Tasks(1).fnc = @(x,y)CP_loss_welsch(x, y,Tasks(1));         

Tasks(2).dims = n;
Tasks(2).pc1 = pc{2};
Tasks(2).pc2 = pc{1};
Tasks(2).norm_pc1 = norm_pc{2};
Tasks(2).norm_pc2 = norm_pc{1};
Tasks(2).resolution = cal_md(norm_pc{1}.Location');   
Tasks(2).kdtree = KDTreeSearcher(Tasks(2).norm_pc2.Location);
Tasks(2).fnc = @(x,y)CP_loss_inliers(x, y,Tasks(2));       
[Tasks.Lb] = deal([l_rot l_trans],[l_rot l_trans]); 
[Tasks.Ub] = deal([u_rot u_trans],[u_rot u_trans]);

iter_times=1;
for ii=1:iter_times
    Ours_data(ii)=MLPCR(Tasks,pop,gen,selection_pressure,rmp,p_il,111,param); % Ours_data(ii)--->>111

  
    p(1).M(1:4,1:4) = eye(4);
    [e_r_Initial,e_t_Initial,e_RMSE_Initial] = calc_errors(Tasks(1).pc1,p(1).M(1:3,1:3),R_real,p(1).M(1:3,4),t_real);
    r_Initial(ii)=e_r_Initial;
    t_Initial(ii)=e_t_Initial;
    RMSE_Initial(ii)=e_RMSE_Initial;

    p(2).M(1:4,1:4) = obtain_p(Ours_data(ii), Tasks, 1, 2);           
    [e_r,e_t,e_RMSE] = calc_errors(Tasks(1).pc1,p(2).M(1:3,1:3),R_real,p(2).M(1:3,4),t_real);
    r_MFEA1(ii)=e_r;
    t_MFEA1(ii)=e_t;
    RMSE_MFEA1(ii)=e_RMSE;
    p(3).M(1:4,1:4) = obtain_p(Ours_data(ii), Tasks, 2, 2);          
    [e_r,e_t,e_RMSE] = calc_errors(Tasks(1).pc1,p(3).M(1:3,1:3),R_real,p(3).M(1:3,4),t_real);
    r_MFEA2(ii)=e_r;
    t_MFEA2(ii)=e_t;
    RMSE_MFEA2(ii)=e_RMSE;
    p(4).M(1:4,1:4) = obtain_p(Ours_data(ii), Tasks, 1, 3);          
    [e_r,e_t,e_RMSE] = calc_errors(Tasks(1).pc1,p(4).M(1:3,1:3),R_real,p(4).M(1:3,4),t_real);
    r_MFEA3(ii)=e_r;
    t_MFEA3(ii)=e_t;
    RMSE_MFEA3(ii)=e_RMSE;
    p(5).M(1:4,1:4) = obtain_p(Ours_data(ii), Tasks, 2, 3);           
    [e_r,e_t,e_RMSE] = calc_errors(Tasks(1).pc1,p(5).M(1:3,1:3),R_real,p(5).M(1:3,4),t_real);
    r_MFEA4(ii)=e_r;
    t_MFEA4(ii)=e_t;
    RMSE_MFEA4(ii)=e_RMSE;
    disp(['Now generation ', num2str(ii)]);
end

disp(['e_r_Initial= ', num2str(mean(r_Initial)), '     e_t_Initial= ', num2str(mean(t_Initial))]);
disp(['e_r_1= ', num2str(mean(r_MFEA1)), '     e_t_1= ', num2str(mean(t_MFEA1))]);
disp(['e_r_2= ', num2str(mean(r_MFEA2)), '     e_t_2= ', num2str(mean(t_MFEA2))]);
disp(['RMSE_Initial= ', num2str(mean(RMSE_Initial))]);
disp(['RMSE_1= ', num2str(mean(RMSE_MFEA1))]);
disp(['RMSE_2= ', num2str(mean(RMSE_MFEA2))]);

time=toc;
figure();
close all;
% figure();plot(Ours_data(ii).EvBestFitness(1,:));figure();plot(Ours_data(ii).EvBestFitness(2,:));hold on;

vslz(Ours_data(ii), Tasks, 1, 2);
vslz(Ours_data(ii), Tasks, 2, 2);






