function data_SOO = SurrogatePSO(Task,pop,gen,CR,reps,prm)    
    tic         
   
    D = Task.dims;
    wmax=0.9; % inertia weight
    wmin=0.4; % inertia weight
    c1=1.49445;
    c2=1.49445;
    ReSurroTrain = [];
    fnceval_calls = zeros(1,reps); 
    calls_per_individual=zeros(1,pop);
    EvBestFitness = zeros(reps,gen);    % best fitness found
    TotalEvaluations=zeros(reps,gen);   % total number of task evaluations so fer
    for rep = 1:reps
%         disp(rep)
        for i = 1 : pop
            population(i) = Particle();
            population(i) = initialize(population(i),D);
        end
        
        for i = 1 : pop
            [population(i),calls_per_individual(i)] = evaluate_SOO(population(i),Task,prm);
            population(i).pbestFitness=population(i).factorial_costs;
            Surro(i,:) = [population(i).pbest population(i).pbestFitness];
        end
%         ReSurroTrain = [ReSurroTrain;Surro];

        fnceval_calls(rep)=fnceval_calls(rep) + sum(calls_per_individual);
        TotalEvaluations(rep,1)=fnceval_calls(rep);    
        [bestobj,bestId]=min([population.factorial_costs]);
        gbest=population(bestId).rnvec;
        EvBestFitness(rep,1) = bestobj;    

        ite=1;
        noImpove=0;
        while ite<=gen
            w1=wmax-(wmax-wmin)*ite/gen;
                for i=1:pop
                    population(i)=velocityUpdate_SOO_new(population(i),gbest,w1,c1,c2);
                end
                for i=1:pop
                    population(i)=positionUpdate(population(i));
                end 

            
            for i=1:pop
                Island(i,:) = population(i).rnvec;
            end 
            
            All_Local = [];
            for k = 1:2
                for i = 1:pop
                    position = population(i).rnvec;
                    x_position = 1 - position;
                    r = rand(1,D);
                    for d = 1:D
                        if r(d) < CR(d)
                            Local(i,d) = x_position(d);
                        else
                            Local(i,d) = position(d);
                        end
                    end
                end
                All_Local = [All_Local;Local];
            end
            pool = [Island;All_Local];
            fitness = SurrogateKNN(1,Surro(:,1:D),Surro(:,end),pool);
            SuP = [pool fitness'];
            [yyy,ra] = sort(SuP(:,end));
            All_R_P = SuP(ra,:);
            TT_pool = pool(ra,:);

            for te_point=1:size(TT_pool,1)
                for tra_point=1:size(TT_pool,1)
                     %calc and store sorted euclidean distances
                     ed(te_point,tra_point)=sqrt(...
                         sum(( TT_pool(te_point,:)- TT_pool(tra_point,:)).^2));
                end
            end

            for i = 1:pop
                population(i).rnvec = All_R_P(1,1:D);
                population(i).factorial_costs = All_R_P(1,end);
                All_R_P = All_R_P(2:end,:);
                [u1 u2] = min(All_R_P(:,1));
                All_R_P = [All_R_P(1:(u2-1),:);All_R_P((u2+1):end,:)];
            end
            
            for i = 1 : pop
                [population(i),calls_per_individual(i)] = evaluate_SOO(population(i),Task,prm); 
                Surro(i,:) = [population(i).rnvec population(i).factorial_costs];
            end
            fnceval_calls(rep)=fnceval_calls(rep) + sum(calls_per_individual);                     
            factorial_cost=zeros(1,pop);
            for i=1:pop
                population(i)=pbestUpdate_SOO(population(i));
            end
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
            EvBestFitness(rep,ite)=bestobj;
            EvBestPosition(ite,:)=gbest; 
%             disp(['SuPSO iteration = ', num2str(ite), ' best fitness value = ', num2str(bestobj)]);                              
            ite=ite+1;
        end     
    end
    data_SOO.wall_clock_time=toc;
    data_SOO.EvBestFitness=EvBestFitness;
    data_SOO.bestInd_data=gbest;
    data_SOO.EvBestPosition=EvBestPosition;
end