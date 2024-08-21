function obj = CP_loss_welsch(x,iter_huber, task)
    angle = x(1:3);
    T = x(4:6);
    R = eul2rotm(angle);    % eul2rotm()   
    M=rigid3d(R', T);  
    tform = affine3d(M.T);
    t_pc1 = pctransform(task.norm_pc1, tform);
    t_pc2 = task.norm_pc2;                      % pcshowpair(t_pc1,t_pc2);

    source = t_pc1.Location;
    target = t_pc2.Location;
    kdtree = task.kdtree;
    [index, dist] = knnsearch(kdtree, source);  %

    obj = sum(1-exp(-   ((dist').^2)./ (2*((task.resolution).^2))   ));


end

