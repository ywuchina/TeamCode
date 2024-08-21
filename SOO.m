function data_SOO = SOO(Task,task_for_comparison_with_SOO,pop,gen,selection_process,p_il, data_MFEA,param)
    tic

    
    if mod(pop,2) ~= 0     
        pop = pop + 1;
    end   
    D = Task.dims;
    options = optimoptions(@fminunc,'Display','off','Algorithm','quasi-newton','MaxIter',5); 
    
    fnceval_calls = 0;
    calls_per_individual=zeros(1,pop);
    EvBestFitness = zeros(1,gen);    
    TotalEvaluations=zeros(1,gen);
        
    if isstruct(data_MFEA)          
%        population = data_MFEA.inital_pop; 
%        disp('data_MFEA is used++++++++++++++++++');
%        pause(0.1);
       population = data_MFEA.inital_pop(1:pop);
    else
        for i = 1 : pop
            population(i) = Chromosome();
            population(i) = initialize(population(i),D);
        end
    end
%     population = data_MFEA.inital_pop;
    parfor i = 1 : pop
        [population(i),calls_per_individual(i)] = evaluate_SOO(population(i),Task,p_il,options,param.iter_huber);
    end
    
    fnceval_calls=fnceval_calls + sum(calls_per_individual);
    TotalEvaluations(1)=fnceval_calls;    
    bestobj=min([population.factorial_costs]);
    EvBestFitness(1) = bestobj;     
    
    generation=1;
%     mu = 2;     %10;
%     sigma = 0.6;  %0.02;
    while generation <= gen
        generation = generation + 1;
        indorder = randperm(pop);
        count=1;
        for i = 1 : pop/2     
            p1 = indorder(i);
            p2 = indorder(i+(pop/2));
            child(count)=Chromosome();
            child(count+1)=Chromosome();
            u = rand(1,D);
            cf = zeros(1,D);
            cf(u<=0.5)=(2*u(u<=0.5)).^(1/(param.mu+1));
            cf(u>0.5)=(2*(1-u(u>0.5))).^(-1/(param.mu+1));
            child(count) = crossover(child(count),population(p1),population(p2),cf);
            child(count+1) = crossover(child(count+1),population(p2),population(p1),cf);
            if rand(1) < 0.1
                child(count)=mutate(child(count),child(count),D,param.sigma);
                child(count+1)=mutate(child(count+1),child(count+1),D,param.sigma);
            end            
            count=count+2;
        end        
        parfor i = 1 : pop            
            [child(i),calls_per_individual(i)] = evaluate_SOO(child(i),Task,p_il,options,param.iter_huber);           
        end      
        
        fnceval_calls=fnceval_calls + sum(calls_per_individual);
        TotalEvaluations(generation)=fnceval_calls;
        
        intpopulation(1:pop)=population;
        intpopulation(pop+1:2*pop)=child;
        [xxx,y]=sort([intpopulation.factorial_costs]);
        intpopulation=intpopulation(y);
        for i = 1:2*pop
            intpopulation(i).scalar_fitness=1/i;
        end
        if intpopulation(1).factorial_costs<=bestobj
            bestobj=intpopulation(1).factorial_costs;
            bestInd_data=intpopulation(1);
        end
        EvBestFitness(generation)=bestobj;
                
        if strcmp(selection_process,'elitist')
            [xxx,y]=sort(-[intpopulation.scalar_fitness]);
            intpopulation=intpopulation(y);
            population=intpopulation(1:pop);            
        elseif strcmp(selection_process,'roulette wheel')
            for i=1:pop
                population(i)=intpopulation(RouletteWheelSelection([intpopulation.scalar_fitness]));
            end    
        end
        if param.iter_huber>param.iter_huber_end
            param.iter_huber=param.iter_huber*param.anneal_rate;
        end
        disp(['SOO ', num2str(task_for_comparison_with_SOO),'  Generation ', num2str(generation), ' best objective = ', num2str(bestobj)])  
    end    
    data_SOO.wall_clock_time=toc;
    data_SOO.EvBestFitness=EvBestFitness;
    data_SOO.bestInd_data=bestInd_data;
    data_SOO.TotalEvaluations=TotalEvaluations;
%     save('data_SOO','data_SOO');
%     figure(task_for_comparison_with_SOO)
%     hold on    
%     plot(EvBestFitness);
%     xlabel('GENERATIONS')
%     ylabel(['TASK ', num2str(task_for_comparison_with_SOO), ' OBJECTIVE'])
%     legend('MFEA','SOO')
end