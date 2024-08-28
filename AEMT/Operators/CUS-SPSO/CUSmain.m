function best = CUSmain(PF,label,prm)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    
    global train_data test_data;
    %% load weka jar, and common interfacing methods.
    curPath = pwd+"\Operators\CUS-SPSO\";
    addpath(genpath(pwd));
    path(path, [curPath+"lib"]);
    path(path, [curPath+'lib\'+'weka']);
    loadWeka(curPath+'lib\'+'weka');
    %% load dataset and parameter setting
    X=PF;
    Y=label+1;
    [k,dim]= size(X);
    m = k;%m???????k??????
    [out] = fsReliefF(X, Y, k, m);
    [out.fList,fs] = sort(out.W,'descend');
    X = X(:,fs);
    [normal_W,PS] = mapminmax(out.fList,0,1); 
    for i = 1:length(normal_W)
       CR(i) = 0.15*sin(normal_W(i)*pi)+0.05; 
    end
    pop=10; % population size 100
    gen=10; % generation count 100
    reps = 3; % repetitions
    threshold = 0.5;
    % divide training set and test set
    [train_data test_data] = divide_data(X,Y);
    Task = benchmark_PSO(train_data,prm);
    % data_SuPSO = SurrogatePSO_backup(Task,pop,gen,CR,reps);
    data_SuPSO = SurrogatePSO(Task,pop,gen,CR,reps,prm);
    IslandSu = data_SuPSO.EvBestPosition(gen,:)>threshold;
%     T_accuracy = Test_Acc(IslandSu)
    Select_F = sum(IslandSu)
    best=IslandSu;

end