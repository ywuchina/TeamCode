function data_MFEA = MLPCR(Tasks,pop,gen,selection_process,rmp,p_il,Old_population,param) 
    
    tic       
    if mod(pop,2) ~= 0
        pop = pop + 1;
    end
    no_of_tasks=length(Tasks);
    if no_of_tasks <= 1
        error('At least 2 tasks required for MFEA');
    end
    D=zeros(1,no_of_tasks);
    for i=1:no_of_tasks
        D(i)=Tasks(i).dims;
    end
    D_multitask=max(D);
   
    options = optimoptions(@fminunc,'Display','off','Algorithm','quasi-newton','MaxIter',5); 
   
    fnceval_calls = 0;   
    calls_per_individual=zeros(1,pop);
    
    EvBestFitness = zeros(no_of_tasks,gen);
    TotalEvaluations=zeros(1,gen);
    bestobj=inf*(ones(1,no_of_tasks));
    
    for i = 1 : pop
        population(i) = Chromosome();
        population(i) = initialize(population(i),D_multitask);
        population(i).skill_factor=0;
    end

    for i = 1 : pop  %parfor
        [population(i),calls_per_individual(i)] = evaluate(population(i),Tasks,p_il,no_of_tasks,options,param.iter_huber);
    end

    fnceval_calls=fnceval_calls + sum(calls_per_individual);
    TotalEvaluations(1)=fnceval_calls;
    factorial_cost=zeros(1,pop); 
    
    for i = 1:no_of_tasks
        for j = 1:pop
            factorial_cost(j)=population(j).factorial_costs(i); 
        end
        [xxx,y]=sort(factorial_cost); 
        population=population(y);
        for j=1:pop
            population(j).factorial_ranks(i)=j; 
        end
        bestobj(i)=population(1).factorial_costs(i);
        EvBestFitness(i,1)=bestobj(i);
        bestInd_data(i)=population(1);
    end
	
    for i=1:pop
        [xxx,yyy]=min(population(i).factorial_ranks);
        x=find(population(i).factorial_ranks == xxx);
        equivalent_skills=length(x);
        if equivalent_skills>1 
            population(i).skill_factor=x(1+round((equivalent_skills-1)*rand(1)));
            tmp=population(i).factorial_costs(population(i).skill_factor);       
            population(i).factorial_costs(1:no_of_tasks)=inf;
            population(i).factorial_costs(population(i).skill_factor)=tmp;
        % else, just set the skill_factor and set the factorial_costs of others as inf
		else
            population(i).skill_factor=yyy;
            tmp=population(i).factorial_costs(population(i).skill_factor);
            population(i).factorial_costs(1:no_of_tasks)=inf;
            population(i).factorial_costs(population(i).skill_factor)=tmp;
        end
    end
    

    generation=0;
    while generation < gen 
        generation = generation + 1;
        indorder = randperm(pop);  
        count=1;
        for i = 1 : pop/2     
            p1 = indorder(i);
            p2 = indorder(i+(pop/2));
            child(count)=Chromosome();
            child(count+1)=Chromosome();
            if (population(p1).skill_factor == population(p2).skill_factor) || (rand(1)<rmp)
                u = rand(1,D_multitask);
                cf = zeros(1,D_multitask);
                cf(u<=0.5)=(2*u(u<=0.5)).^(1/(param.mu+1)); 
                cf(u>0.5)=(2*(1-u(u>0.5))).^(-1/(param.mu+1));
                child(count) = crossover(child(count),population(p1),population(p2),cf);
                child(count+1) = crossover(child(count+1),population(p2),population(p1),cf);
                                
                sf1=1+round(rand(1));
                sf2=1+round(rand(1));
                if sf1 == 1
                    child(count).skill_factor=population(p1).skill_factor;
                else
                    child(count).skill_factor=population(p2).skill_factor;
                end
                if sf2 == 1
                    child(count+1).skill_factor=population(p1).skill_factor;
                else
                    child(count+1).skill_factor=population(p2).skill_factor;
                end
            else   
                child(count)=mutate(child(count),population(p1),D_multitask,param.sigma);
                child(count).skill_factor=population(p1).skill_factor;
                child(count+1)=mutate(child(count+1),population(p2),D_multitask,param.sigma);
                child(count+1).skill_factor=population(p2).skill_factor;
            end
            count=count+2;
        end
        
        for i = 1 : pop  %parfor
            [child(i),calls_per_individual(i)] = evaluate(child(i),Tasks,p_il,no_of_tasks,options,param.iter_huber);           
        end         
        
        fnceval_calls=fnceval_calls + sum(calls_per_individual);
        TotalEvaluations(generation)=fnceval_calls;
        
        intpopulation(1:pop)=population;
        intpopulation(pop+1:2*pop)=child;
       
        factorial_cost=zeros(1,2*pop);  %1X2pop 
        for i = 1:no_of_tasks
            for j = 1:2*pop
                factorial_cost(j)=intpopulation(j).factorial_costs(i);
            end
            [xxx,y]=sort(factorial_cost);    
            intpopulation=intpopulation(y);  
            for j=1:2*pop
                intpopulation(j).factorial_ranks(i)=j; 
            end
           
            if intpopulation(1).factorial_costs(i)<=bestobj(i)
                bestobj(i)=intpopulation(1).factorial_costs(i);
                bestInd_data(i)=intpopulation(1);
            end
            EvBestFitness(i,generation)=bestobj(i);            
        end
        for i=1:2*pop
            [xxx,yyy]=min(intpopulation(i).factorial_ranks);
            intpopulation(i).skill_factor=yyy;
            intpopulation(i).scalar_fitness=1/xxx;
        end

        if strcmp(selection_process,'elitist')
            [xxx,y]=sort(-[intpopulation.scalar_fitness]); 
            intpopulation=intpopulation(y);
            population=intpopulation(1:pop);            
        elseif strcmp(selection_process,'roulette wheel')
            for i=1:no_of_tasks
                skill_group(i).individuals=intpopulation([intpopulation.skill_factor]==i);
            end
            count=0;
            while count<pop
                count=count+1;
                skill=mod(count,no_of_tasks)+1;
                population(count)=skill_group(skill).individuals(RouletteWheelSelection([skill_group(skill).individuals.scalar_fitness]));
            end     
        end
        
        if param.iter_huber>param.iter_huber_end
            param.iter_huber=param.iter_huber*param.anneal_rate;
        end

        % disp(['MFEA Generation =         ', num2str(generation), '          best factorial costs =                ', num2str(bestobj)]);
    end 
    data_MFEA.wall_clock_time=toc;
    data_MFEA.EvBestFitness=EvBestFitness;
    data_MFEA.bestInd_data=bestInd_data;
    data_MFEA.TotalEvaluations=TotalEvaluations;

    data_MFEA.OLDbestInd_data = bestInd_data;  
    [data_MFEA.bestInd_data(1, 1),~] = evaluate(data_MFEA.bestInd_data(1, 1),Tasks,1,no_of_tasks,options,param.iter_huber);           
    [data_MFEA.bestInd_data(1, 2),~] = evaluate(data_MFEA.bestInd_data(1, 2),Tasks,1,no_of_tasks,options,param.iter_huber);           

    
    
    
    
   
end