function data_SOO = PSO(Task,pop,gen,p_il,reps)
%SOEA function: implementation of SOEA algorithm
    clc    
    tic         
    if mod(pop,2) ~= 0
        pop = pop + 1;
    end   
    D = Task.dims;
    options = optimoptions(@fminunc,'Display','off','Algorithm','quasi-newton','MaxIter',2);  % settings for individual learning
    
    wmax=0.9; % inertia weight
    wmin=0.4; % inertia weight
    c1=0.2;
    c2=0.2;
    w11=1000;
    c11=1000;
    c22=1000; 
    
    fnceval_calls = zeros(1,reps); 
    calls_per_individual=zeros(1,pop);
    EvBestFitness = zeros(reps,gen);    % best fitness found
    TotalEvaluations=zeros(reps,gen);   % total number of task evaluations so fer
    for rep = 1:reps   
        disp(rep)
        for i = 1 : pop
            population(i) = Particle();
            population(i) = initialize(population(i),D);
        end
        
        for i = 1 : pop
            [population(i),calls_per_individual(i)] = evaluate_SOO(population(i),Task,p_il,options);
            population(i).pbestFitness=population(i).factorial_costs;
        end

        fnceval_calls(rep)=fnceval_calls(rep) + sum(calls_per_individual);
        TotalEvaluations(rep,1)=fnceval_calls(rep);    
        [bestobj,bestId]=min([population.factorial_costs]);
        gbest=population(bestId).rnvec;
        EvBestFitness(rep,1) = bestobj;    

        ite=1;
        noImpove=0;
        while ite<=gen
            w1=wmax-(wmax-wmin)*ite/gen;
            if ~mod(ite,10)&&noImpove>=20
                for i=1:pop
                    population(i)=velocityUpdate_SOO(population(i),gbest,w11,c11,c22);
                end
            else             
                for i=1:pop
                    population(i)=velocityUpdate_SOO(population(i),gbest,w1,c1,c2);
                end
            end
            for i=1:pop
                population(i)=positionUpdate(population(i));
            end          
            for i=1:pop
                population(i)=pbestUpdate_SOO(population(i));
            end      
            
            for i = 1:pop            
                [population(i),calls_per_individual(i)] = evaluate_SOO(population(i),Task,p_il,options);         
            end             
            fnceval_calls(rep)=fnceval_calls(rep) + sum(calls_per_individual);                     
            
            factorial_cost=zeros(1,pop);
            for j = 1:pop
            	factorial_cost(j)=population(j).factorial_costs;
            end
        	[xxx,y]=sort(factorial_cost);
            population=population(y);

            if population(1).factorial_costs<=bestobj
                bestobj=population(1).factorial_costs;
              	gbest=population(1).rnvec;
                bestInd_data(rep)=population(1);
                noImpove=0;
            else
                noImpove=noImpove+1;                
            end        
            EvBestFitness(rep,ite+1)=bestobj;
            disp(['PSO iteration = ', num2str(ite), ' best factorial costs = ', num2str(bestobj)]);                              
            ite=ite+1;
        end     
    end
    data_SOO.wall_clock_time=toc;
    data_SOO.EvBestFitness=EvBestFitness;
    data_SOO.bestInd_data=bestInd_data;
    data_SOO.TotalEvaluations=TotalEvaluations;
end