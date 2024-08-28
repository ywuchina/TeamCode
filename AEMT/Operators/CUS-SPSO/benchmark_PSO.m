function Tasks = benchmark_PSO(X,prm)
%BENCHMARK function
%   Output:
%   - Tasks: benchmark problem set

dim = size(X,2)-1;
Tasks.prm=prm;
Tasks.dims = dim;
Tasks.fnc = @(fits)eva4(X,prm);
Tasks.Lb=0*ones(1,dim);
Tasks.Ub=1*ones(1,dim);
end





