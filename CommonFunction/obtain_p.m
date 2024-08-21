function [MM] = obtain_p(data_MorS, Tasks, i, flag)
    if flag == 1       
        vec = data_MorS.bestInd_data(1,1).rnvec;
        str = 'SOO';
    elseif flag ==2    
        vec = data_MorS.bestInd_data(1,i).rnvec;
        str = 'MFEA';
    elseif flag ==3     
        vec = data_MorS.OLDbestInd_data(1,i).rnvec;
        str = 'AG';
    end
    
    minrange = Tasks(i).Lb;
    maxrange = Tasks(i).Ub;
    y=maxrange-minrange;        
    vars = y.*vec + minrange;   
    
    angle = vars(1:3);
    T = vars(4:6);
    R = eul2rotm(angle);
    M=rigid3d(R', T);  
    MM=M.T';
end