% This MATLAB R2014b code is for EVOLUTIONARY MULTITASKING across minimization problems. 
% For maximization problems, multiply objective function by -1.

% Settings of simulated binary crossover (SBX) in this code is Pc = 1, 
% and probability of variable sawpping = 0. 

% For suggestions please contact: Bingshui Da (Email: da0002ui@e.ntu.edu.sg)
clear;
clc;
close all;

% load('G:\featurefile\cloudbintrans.mat');
% 
% mat1 = load('G:\featurefile\cloudbin1.mat');
% 
% matpc = load('G:\pcdate\cloudbinpc.mat');
% 
% names1 = fieldnames(mat1);
% f1 = mat1.(names1{1}); 
% 
% names3 = fieldnames(matpc);
% f3 = matpc.(names2{1});


f1 = load('./Data/feature/AllFeature.mat').pallF; %特征
f3 = load('./Data/PC/p.mat').p; %点云
load('./Data/PC/7-scenes-stairs.mat'); % 变换
load('./Data/feature/LEN.mat');


% feat1 = f1(:,50:401);
% feat2 = f2(:,50:401);
% tform = tran{2,4};
% ts11 = shotsum3(feat1);z
% ts12 = shotsum3(feat2);
% ts21 = shotsum5(feat1);
% ts22 = shotsum5(feat2);
%% Calling the solvers
% For large population sizes, consider using the Parallel Computing Toolbox
% of MATLAB.
% Else, program can be slow.
pop_M=50; % population size 100
%pop_S = pop_M;
gen=30; % generation count 1000 

p_il = 0; % probability of individual learning (BFGA quasi-Newton Algorithm) --> Indiviudal Learning is an IMPORTANT component of the MFEA.
rmp=0.3; % random mating probability
reps = 1; % repetitions 20
% index = 0;
% Tasks = benchmark(index);
% dim1 = 517;
% dim2 = 96;
% dim3 = 140;
% dim4 = 252;
% dim5 = 29;
dim1 = sum(LEN);
dim2 = LEN(1,1);
dim3 = LEN(1,2);
dim4 = LEN(1,3);
dim5 = LEN(1,4);
bestft = [];
Timing = zeros(1,10);
ind = 1;
% for p = 1:19 
%     for q = p+1:20
for k = 1:10
for p = 1:1
    for q = 2:2
            t=[GrtR{q},GrtT{q}];
            t=[t;0 0 0 1];
            t=t';
            T1=t;
            t=[GrtR{p},GrtT{p}];
            t=[t;0 0 0 1];
            t=t';
            T2=t;
            tform=affine3d(round(T1/T2,20));
%             tform = tran{p,q};

%             ts11=f1{p,q}(:,:);
%             ts12=f1{p+3,q+3}(:,:);
            ts11=f1{p}(:,:);
            ts12=f1{q}(:,:);
            Tasks(1).dims = dim1;    % dimensionality of Task 1
            Tasks(1).Lb=zeros(1,dim1);   % Upper bound of Task 1
            Tasks(1).Ub=ones(1,dim1);    % Lower bound of Task 1
            Tasks(1).feat1 = ts11;
            Tasks(1).feat2 = ts12;
%             Tasks(1).tarCloud = f3(p,q);
            Tasks(1).tarCloud = f3(p);
%             Tasks(1).srcCloud = f3(p+3,q+3);
            Tasks(1).srcCloud = f3(q);
            Tasks(1).tform = tform;
            Tasks(1).fnc = @(x)fitness(x,Tasks(1));
            
%             ts21=f1{p,q}(1:dim2);
%             ts22=f1{p+3,q+3}(1:dim2);
            ts21=f1{p}(:,1:dim2);
            ts22=f1{q}(:,1:dim2);
            Tasks(2).dims = dim2;    
            Tasks(2).Lb=zeros(1,dim2);    
            Tasks(2).Ub=ones(1,dim2);     
            Tasks(2).feat1 = ts21;
            Tasks(2).feat2 = ts22;
%             Tasks(2).tarCloud = f3(p,q);
%             Tasks(2).srcCloud = f3(p+3,q+3);
            Tasks(2).tarCloud = f3(p);
            Tasks(2).srcCloud = f3(q);

            Tasks(2).tform = tform;
            Tasks(2).fnc = @(x)fitness(x,Tasks(2));
            
%             ts31=f1{p,q}(dim2+1:dim3);
%             ts32=f1{p+3,q+3}(dim2+1:dim3);
            ts31=f1{p}(:,dim2+1:dim2+dim3);
            ts32=f1{q}(:,dim2+1:dim2+dim3);
            Tasks(3).dims = dim3;    
            Tasks(3).Lb=zeros(1,dim3);    
            Tasks(3).Ub=ones(1,dim3);     
            Tasks(3).feat1 = ts31;
            Tasks(3).feat2 = ts32;
%             Tasks(3).tarCloud = f3(p,q);
%             Tasks(3).srcCloud = f3(p+3,q+3);
            Tasks(3).tarCloud = f3(p);
            Tasks(3).srcCloud = f3(q);
            Tasks(3).tform = tform;
            Tasks(3).fnc = @(x)fitness(x,Tasks(3));

%             ts41=f1{p,q}(dim3+1:dim4);
%             ts42=f1{p+3,q+3}(dim3+1:dim4);
            ts41=f1{p}(:,dim2+dim3+1:dim2+dim3+dim4);
            ts42=f1{q}(:,dim2+dim3+1:dim2+dim3+dim4);
            Tasks(4).dims = dim4;    
            Tasks(4).Lb=zeros(1,dim4);    
            Tasks(4).Ub=ones(1,dim4);     
            Tasks(4).feat1 = ts41;
            Tasks(4).feat2 = ts42;
%             Tasks(4).tarCloud = f3(p,q);
%             Tasks(4).srcCloud = f3(p+3,q+3);
            Tasks(4).tarCloud = f3(p);
            Tasks(4).srcCloud = f3(q);
            Tasks(4).tform = tform;
            Tasks(4).fnc = @(x)fitness(x,Tasks(4));

%             ts51=f1{p,q}(dim4+1:dim5);
%             ts52=f1{p+3,q+3}(dim4+1:dim5);
            ts51=f1{p}(:,dim2+dim3+dim4+1:dim2+dim3+dim4+dim5);
            ts52=f1{q}(:,dim2+dim3+dim4+1:dim2+dim3+dim4+dim5);
            Tasks(5).dims = dim5;    
            Tasks(5).Lb=zeros(1,dim5);    
            Tasks(5).Ub=ones(1,dim5);     
            Tasks(5).feat1 = ts41;
            Tasks(5).feat2 = ts42;
%             Tasks(5).tarCloud = f3(p,q);
%             Tasks(5).srcCloud = f3(p+3,q+3);
            Tasks(5).tarCloud = f3(p);
            Tasks(5).srcCloud = f3(q);
            Tasks(5).tform = tform;
            Tasks(5).fnc = @(x)fitness(x,Tasks(5));

        tic;
        data_MFPSO=MFPSO(Tasks,pop_M,gen,rmp,p_il,reps);
        toc;
        Timing(1,k) = toc;
        
        best(ind,:)=data_MFPSO.bestInd_data(1).rnvec;
        ind = ind + 1;
    end
end
end
disp(['运行时间: ',num2str(Timing)]);
%%
function feat = shotsum3(feat)   

    featsum = [];
    for i = 1:32
        shot{i} = feat(:,11*(i-1)+1:11*i);
        sumf1 = sum(shot{i}(:,1:4),2);
        sumf2 = sum(shot{i}(:,5:8),2);
        sumf3 = sum(shot{i}(:,9:11),2);
        featsum = [featsum,sumf1,sumf2,sumf3];
    end 
    feat = featsum;
end

function feat = shotsum5(feat)   

    featsum = [];
    for i = 1:32
        shot{i} = feat(:,11*(i-1)+1:11*i);
        sumf1 = sum(shot{i}(:,1:2),2);
        sumf2 = sum(shot{i}(:,3:4),2);
        sumf3 = sum(shot{i}(:,5:6),2);
        sumf4 = sum(shot{i}(:,7:8),2);
        sumf5 = sum(shot{i}(:,9:11),2);
        featsum = [featsum,sumf1,sumf2,sumf3,sumf4,sumf5];
    end 
    feat = featsum;
end

    % task_for_comparison_with_SOO = 1;
    % data_SOO_1(index)=PSO(Tasks(task_for_comparison_with_SOO),pop_S,gen,p_il,reps);   
% 
%     task_for_comparison_with_SOO = 2;
%     data_SOO_2(index)=PSO(Tasks(task_for_comparison_with_SOO),pop_S,gen,p_il,reps);     


%save('result.mat','data_MFPSO', 'data_SOO_1', 'data_SOO_2');

