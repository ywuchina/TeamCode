classdef Particle    
    properties
        rnvec; % (genotype)--> decode to find design variables --> (phenotype)
        pbest;
        pbestFitness;
        velocity;
        factorial_costs;
        factorial_ranks;
        scalar_fitness;
        skill_factor;
    end    
    methods        
        function object = initialize(object,D)
            object.rnvec = rand(1,D);
            object.pbest = object.rnvec;
            object.velocity = 0.1*object.pbest;
        end
        
        function object = initialize_new(object,D,threshold,number_initial)
            for i = 1:number_initial
                value = (1-(0.9-threshold)*(i/number_initial));
                object.rnvec(1,i) = value*rand;
            end
            for i = (number_initial+1):D
                value = threshold+0.1;
                object.rnvec(1,i) = value*rand;
            end
            object.pbest = object.rnvec;
            object.velocity = 0.1*object.pbest;
        end
        
        function [object,calls] = evaluate(object,Tasks,p_il,no_of_tasks,options)     
            if object.skill_factor == 0
                calls=0;
                for i = 1:no_of_tasks
                    [object.factorial_costs(i),xxx,funcCount]=fnceval(Tasks(i),object.rnvec,i,p_il,options);
                    calls = calls + funcCount;
                end
            else
                object.factorial_costs(1:no_of_tasks)=inf;
                for i = 1:no_of_tasks
                    if object.skill_factor == i
                        [object.factorial_costs(object.skill_factor),object.rnvec,funcCount]=fnceval(Tasks(object.skill_factor),object.rnvec,object.skill_factor,p_il,options);
                        calls = funcCount;
                        break;
                    end
                end
            end
        end
        
        function [object,calls] = evaluate_SOO(object,Task,prm)   
            [object.factorial_costs,object.rnvec,funcCount]=fnceval(Task,object.rnvec,prm);
            calls = funcCount;
        end
        
        % position update
        function object=positionUpdate(object)                       
            object.rnvec=object.rnvec+object.velocity;
            object.rnvec(object.rnvec>1)=1;
            object.rnvec(object.rnvec<0)=0;           
        end
        
       function object=positionUpdate_new(object,threshold,number_initial)                       
            object.rnvec=object.rnvec+object.velocity;
            dim = length(object.rnvec);
            for i = 1:number_initial
                object.rnvec(object.rnvec(1,i)>(1-(0.9-threshold)*(i/dim))) = 1-(0.9-threshold)*(i/dim);
            end
            for i = (number_initial+1):dim
                object.rnvec(object.rnvec(1,i)>(threshold+0.1)) = (threshold+0.1);
            end
            object.rnvec(object.rnvec<0)=0;           
        end
        
        % pbest update
        function object=pbestUpdate(object)
              if object.factorial_costs(object.skill_factor)<object.pbestFitness
                  object.pbestFitness=object.factorial_costs(object.skill_factor);
                  object.pbest=object.rnvec;
              end                   
        end 
 
        function object=pbestUpdate_SOO(object)
              if object.factorial_costs<object.pbestFitness
                  object.pbestFitness=object.factorial_costs;
                  object.pbest=object.rnvec;
              end                   
        end 
%                 % velocity update
%         function object=velocityUpdate_new(object,population,value,gbest,rmp,w1,c1,c2,c3,Tasks,p_il,options)
%             len=length(object.velocity);
% w1*object.velocity+c1*rand(1,len).*(gv-object.rnvec);
% %             if rand()<0.5
% %                 object.skill_factor=2/object.skill_factor;
% %             end
%             if rand()<rmp
%                 %original            %new DE is utilized to breed promising gv
% %             r1 = round(rand(1,1)*((length(population))-1))+1;
% %             r2 = round(rand(1,1)*((length(population))-1))+1;
% %             while r1==r2||r1==value||r2==value
% %                  r1=round(rand(1,1)*((length(population))-1))+1;  
% %                  r2=round(rand(1,1)*((length(population))-1))+1; 
% %             end
% %             v = gbest(object.skill_factor,:)+gbest(2/object.skill_factor,:)+0.5*(population(r1).pbest-population(r2).pbest);
% %             jrand = round(rand(1,1)*(len-1))+1;
% %             for d=1:len
% %                 r=rand(1,1);
% %                 if r<0.025||d==jrand
% %                 u(1,d)=v(1,d);
% %                 else
% %                 u(1,d)=object.pbest(1,d);
% %                 end
% %                 if u(1,d)>(1-(0.9-0.6)*(d/len))
% %                    u(1,d)=(1-(0.9-0.6)*(d/len));
% %                 end
% %                 if u(1,d)<0
% %                    u(1,d)=0;
% %                 end
% %             end
% %             [objective_value,xxx,funcCount] = fnceval(Tasks(1),u,p_il,options);
% %             if objective_value<object.factorial_costs
% %                 gv = u;
% %             else
% %                 gv = object.pbest;
% %             end
% %             object.velocity= 
%                 object.velocity=w1*object.velocity+c1*rand(1,len).*(object.pbest-object.rnvec)...
%                 	+c2*rand(1,len).*(gbest(object.skill_factor,:)-object.rnvec)+c3*rand(1,len).*(gbest(2/object.skill_factor,:)-object.rnvec);
%                 if rand()<0.5
%                     object.skill_factor=2/object.skill_factor;
%                 end
%             else
%                 object.velocity=w1*object.velocity+c1*rand(1,len).*(object.pbest-object.rnvec)...
%                     +c2*rand(1,len).*(gbest(object.skill_factor,:)-object.rnvec);
%             end        
%         end
        % velocity update
        function object=velocityUpdate(object,fitness,gbest,rmp,w1,c1,c2,c3,no_of_tasks)
            len=length(object.velocity);
            nsc = tournamentSelection(fitness,object.skill_factor);
%             nsc = randperm(no_of_tasks,1);
            if nsc == object.skill_factor
                nsc = tournamentSelection(fitness,object.skill_factor);
            end
            if rand()<rmp
                %original
                cr = rand(1,len);
                mean_p = cr.*gbest(object.skill_factor,:) + (1-cr).*gbest(nsc,:);
                object.velocity=w1*object.velocity+c1*rand(1,len).*(object.pbest-object.rnvec)...
                    +c2*rand(1,len).*(mean_p-object.rnvec);
%                 object.velocity=w1*object.velocity+c1*rand(1,len).*(object.pbest-object.rnvec)...
%                 	+c2*rand(1,len).*(gbest(object.skill_factor,:)-object.rnvec)+c3*rand(1,len).*(gbest(nsc,:)-object.rnvec);
%                 if rand()<0.5
%                     object.skill_factor=nsc;
%                 end
            else
                object.velocity=w1*object.velocity+c1*rand(1,len).*(object.pbest-object.rnvec)...
                    +c2*rand(1,len).*(gbest(object.skill_factor,:)-object.rnvec);
            end        
        end
        
        function object=velocityUpdate_escape(object,gbest,w1,c1,c2,no_of_tasks)
            len=length(object.velocity);
            mean_total = zeros(1,len);
            total = zeros(1,len);
            for k = 1:no_of_tasks
                total = total + gbest(k,:);
            end
            mean_total = total./no_of_tasks;
        	object.velocity=w1*object.velocity+c1*rand(1,len).*(object.pbest-object.rnvec)...
            	+c2*rand(1,len).*(mean_total-object.rnvec);
        end
        
%         % velocity update
%         function object=velocityUpdate(object,gbest,rmp,w1,c1,c2,c3)
%             len=length(object.velocity);
%             if rand()<rmp
%                 %original
%                 object.velocity=w1*object.velocity+c1*rand(1,len).*(object.pbest-object.rnvec)...
%                 	+c2*rand(1,len).*(gbest(object.skill_factor,:)-object.rnvec)+c3*rand(1,len).*(gbest(2/object.skill_factor,:)-object.rnvec);
%                 if rand()<0.5
%                     object.skill_factor=2/object.skill_factor;
%                 end
%             else
%                 object.velocity=w1*object.velocity+c1*rand(1,len).*(object.pbest-object.rnvec)...
%                     +c2*rand(1,len).*(gbest(object.skill_factor,:)-object.rnvec);
%             end        
%         end
        
        function object=velocityUpdate_SOO(object,gbest,w1,c1,c2)
            len=length(object.velocity);
        	object.velocity=w1*object.velocity+c1*rand(1,len).*(object.pbest-object.rnvec)...
            	+c2*rand(1,len).*(gbest-object.rnvec);
        end
        
        function object=velocityUpdate_SOO_new(object,gbest,w1,c1,c2)
            len=length(object.velocity);
        	object.velocity=w1*object.velocity+c1*rand(1,len).*(object.pbest-object.rnvec)...
            	+c2*rand(1,len).*(gbest-object.rnvec);
        end
        
    end
end