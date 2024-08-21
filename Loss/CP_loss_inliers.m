
function obj = CP_loss_inliers(x,iter_huber, task)
    angle = x(1:3);
    T = x(4:6);
    R = eul2rotm(angle);                                      
    M=rigid3d(R', T);  
    tform = affine3d(M.T);
    t_pc1 = pctransform(task.norm_pc1, tform);
    t_pc2 = task.norm_pc2;
    
    source = t_pc1.Location;
    target = t_pc2.Location;                                      % pcshowpair(t_pc1,t_pc2)
    [index, dist] = knnsearch(task.kdtree, source);               
    
    inlier=sum(dist<=task.resolution);
    obj = 1-inlier/size(dist,1);
    
               
end

