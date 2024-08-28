classdef TASK    
    %This class contains all task information and needs to be initialized
    %by initTASK.
    properties
        dim;                                % 任务维度
        space;                        % 搜索空间，长度为总维度，1代表选择该维特征，0代表不选择该维特征
    end    
    methods        
        function object = initTASK(object,d,s)
            object.dim=d;
            object.space=s;
        end  
    end
end