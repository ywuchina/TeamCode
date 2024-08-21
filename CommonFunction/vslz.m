function [] = vslz(data_MorS, Tasks, i, flag)
    if flag == 1        
        vec = data_MorS.bestInd_data(1,1).rnvec;
        str = 'SOO';
    elseif flag ==2    
        vec = data_MorS.bestInd_data(1,i).rnvec;
        str = 'MFEA';
    end
    
    minrange = Tasks(i).Lb;
    maxrange = Tasks(i).Ub;
    y=maxrange-minrange;        
    vars = y.*vec + minrange;   
    angle = vars(1:3);
    T = vars(4:6);
    R = eul2rotm(angle);
    M=rigid3d(R', T);  
    tform = affine3d(M.T);
    t_pc1 = pctransform(Tasks(i).pc1, tform);   % pc1   norm_pc1 
    figure
%     pcshowpair(t_pc1, Tasks(i).norm_pc2);            % pc2   norm_pc2
%     title([str, ',task', num2str(i)]);
   

    model = Tasks(i).pc2.Location';
    data = t_pc1.Location';
    plot3(model(1,:),model(2,:),model(3,:),'g.',data(1,:),data(2,:),data(3,:),'b.');
    axis equal;    
    axis off;

end