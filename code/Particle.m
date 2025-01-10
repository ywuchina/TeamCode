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
        
        function [object,calls] = evaluate(object,Tasks,p_il,no_of_tasks,options)     
            if object.skill_factor == 0
                calls=0;
                for i = 1:no_of_tasks
                    [object.factorial_costs(i),xxx,funcCount]=fnceval(Tasks(i),object.rnvec,p_il,options);
                    calls = calls + funcCount;
                end
            else
                object.factorial_costs(1:no_of_tasks)=inf;
                for i = 1:no_of_tasks
                    if object.skill_factor == i
                        [object.factorial_costs(object.skill_factor),object.rnvec,funcCount]=fnceval(Tasks(object.skill_factor),object.rnvec,p_il,options);
                        calls = funcCount;
                        break;
                    end
                end
            end
        end
        
        function [object,calls] = evaluate_SOO(object,Task,p_il,options)   
            [object.factorial_costs,object.rnvec,funcCount]=fnceval(Task,object.rnvec,p_il,options);
            calls = funcCount;
        end
        
        % position update
        function object=positionUpdate(object)
            object.rnvec = object.rnvec(1:673);
            object.rnvec=object.rnvec+object.velocity;
            object.rnvec(object.rnvec>1)=1;
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
        
        % velocity update
        function object=velocityUpdate(object,gbest,rmp,w1,c1,c2,c3)
            len=length(object.velocity);
            if rand()<rmp
                object.velocity=w1*object.velocity+c1*rand(1,len).*(object.pbest-object.rnvec(1:673))...
                	+c2*rand(1,len).*(gbest(object.skill_factor,:)-object.rnvec(1:673))+c3*rand(1,len).*(gbest(object.skill_factor,:)-object.rnvec(1:673));

%                 if rand()<0.5
%                     object.skill_factor=2/object.skill_factor;
%                 end
            else
%                 disp(['1:']);
%                 length(object.velocity)
%                 disp(['2:']);
%                 length(object.pbest)
%                 disp(['3:']);
%                 length(object.rnvec)
                
                object.velocity=w1*object.velocity+c1*rand(1,len).*(object.pbest-object.rnvec(1:673))...
                    +c2*rand(1,len).*(gbest(object.skill_factor,:)-object.rnvec(1:673));
            end        
        end
        
        function object=velocityUpdate_SOO(object,gbest,w1,c1,c2)
            len=length(object.velocity);
        	object.velocity=w1*object.velocity+c1*rand(1,len).*(object.pbest-object.rnvec)...
            	+c2*rand(1,len).*(gbest-object.rnvec);
        end
        
    end
end