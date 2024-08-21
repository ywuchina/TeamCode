
function [objective,rnvec,funcCount] = fnceval(Task,rnvec,p_il,options,iter_huber)
    d = Task.dims;
    nvars = rnvec(1:d);            
    minrange = Task.Lb(1:d);
    maxrange = Task.Ub(1:d);
    y=maxrange-minrange;
    vars = y.*nvars + minrange;    
    if rand(1)<=p_il           
        angle = vars(1:3);        
        T = vars(4:6);
        R = eul2rotm(angle);            
        M=rigid3d(R', T);  
        tform = affine3d(M.T);
        t_pc1 = pctransform(Task.norm_pc1, tform);
        t_pc2 = Task.norm_pc2;                                          % pcshowpair(t_pc1,t_pc2);
        [RotMat2, TransVec2] = RSICPEC(t_pc1,t_pc2);
        M_new = [RotMat2,TransVec2;0 0 0 1]*((M.T)');
        R_new = M_new(1:3,1:3);
        T_new = M_new(1:3,4);                              % figure();pcshowpair(pointCloud(model'),pointCloud(dataOut2'))
        x = rotm2eul(R_new);
        x = [x,T_new'];

        nvars= (x-minrange)./y;    
        m_nvars=nvars;
        m_nvars(nvars<0)=0;
        m_nvars(nvars>1)=1;
        if ~isempty(m_nvars~=nvars) 
            nvars=m_nvars;
            x=y.*nvars + minrange; 
            objective=Task.fnc(x,iter_huber);
        end
        rnvec(1:d)=nvars;
        funcCount=1;
    else
        x=vars;
        objective=Task.fnc(x,iter_huber);
        funcCount=1;
    end
end
