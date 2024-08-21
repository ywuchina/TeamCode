
classdef Chromosome    
    properties
        rnvec; 
        factorial_costs;
        factorial_ranks;
        scalar_fitness;
        skill_factor;
    end    
    methods        
        function object = initialize(object,D)            
            object.rnvec = rand(1,D);
            
        end
        function population = initializeByCube(population,pop_size,D_multitask)            
            population(1).rnvec(1,1) = 2*rand()-1;
            for i=2:D_multitask
                population(1).rnvec(1,i) = 4*(population(1).rnvec(1,i-1).^3)-3*(population(1).rnvec(1,i-1));
            end
            for i=2:pop_size
                population(i).rnvec(1,:)=4*(population(i-1).rnvec(1,:).^3)-3*(population(i-1).rnvec(1,:));
            end
            for i=1:pop_size
                population(i).rnvec = (population(i).rnvec+1)/2;
            end
        end
        function [object,calls] = evaluate(object,Tasks,p_il,no_of_tasks,options,iter_huber) 
            if object.skill_factor == 0
                calls=0;
                for i = 1:no_of_tasks
                    [object.factorial_costs(i),xxx,funcCount]=fnceval(Tasks(i),object.rnvec,p_il,options,iter_huber);
                    calls = calls + funcCount;
                end
            else
                object.factorial_costs(1:no_of_tasks)=inf; 
                for i = 1:no_of_tasks
                    if object.skill_factor == i 
                        [object.factorial_costs(object.skill_factor),object.rnvec,funcCount]=fnceval(Tasks(object.skill_factor),object.rnvec,p_il,options,iter_huber);
                        calls = funcCount;
                        break;
                    end
                end
            end
        end

        function [object,calls] = evaluate_SOO(object,Task,p_il,options,iter_huber)   
            [object.factorial_costs,object.rnvec,funcCount]=fnceval(Task,object.rnvec,p_il,options,iter_huber);
            calls = funcCount;
        end
      
        function object=crossover(object,p1,p2,cf) % SBX 模拟二进制交叉，交叉前后解码实数值的平均相等
            object.rnvec=0.5*((1+cf).*p1.rnvec + (1-cf).*p2.rnvec);
            object.rnvec(object.rnvec>1)=1;%维持rnvec在0~1之间
            object.rnvec(object.rnvec<0)=0;
        end
        
        function object=mutate(object,p,D,sigma)
            rvec=normrnd(0,sigma,[1,D]); % 产生随机数，均值0，标准差sigma，大小是1XD
            object.rnvec=p.rnvec+rvec;   % 累加到原来的基因上
            object.rnvec(object.rnvec>1)=1;%维持rnvec在0~1之间
            object.rnvec(object.rnvec<0)=0;
        end   
        
        %dhq写的分段交叉
        function object=centercrossover(object,p1,p2,cf) % 效果不好、分析是交叉变化太小
            p1.rnvec(:,1:3) = p2.rnvec(:,1:3);
            object.rnvec = p1.rnvec;        
        end
        

        %%  polynomial mutation多项式变异  sigma=2 mu=5   sigma=10 mu=10（GMFEA）
%         function object=mutate(object,p,dim,mum)
%             rnvec_temp=p.rnvec;
%             for i=1:dim
%                 if rand(1)<1/dim
%                     u=rand(1);
%                     if u <= 0.5
%                         del=(2*u)^(1/(1+mum)) - 1;
%                         rnvec_temp(i)=p.rnvec(i) + del*(p.rnvec(i));
%                     else
%                         del= 1 - (2*(1-u))^(1/(1+mum));
%                         rnvec_temp(i)=p.rnvec(i) + del*(1-p.rnvec(i));
%                     end
%                 end
%             end  
%             object.rnvec = rnvec_temp;   
%             object.rnvec(object.rnvec>1)=1;
%             object.rnvec(object.rnvec<0)=0;
%         end
    end
end