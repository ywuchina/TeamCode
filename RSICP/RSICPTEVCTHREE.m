function [T, count] = RSICP(SP,TP,SN,TN)

T = eye(4);
Btree = KDTreeSearcher(TP');
[~, dist] = knnsearch(Btree,TP','k',7);                     % 
dist = dist(:,2:7);                                         % 
u2 = median(median(dist,2))/(3*sqrt(3));                    % 
[p1,p2,n1,n2,r] = match_points(SP,TP,SN,TN,Btree);          % 
u1 = 3*median(r);                                           % 

stop1 = 0; count = 0;
while(~stop1)
    for i=1:100
        w = weight(r,u1);                                   % 1x15458 double
        T0 = symm_po_pl(p1,p2,n1,n2,w);                     % 4x4 double
        T = T0*T;                                           %   
        p12 = T*[SP;ones(1,size(SP,2))]; p1 = p12(1:3,:);   % 
        n1 = T(1:3,1:3)*SN;                                 % 
        [p1,p2,n1,n2,r] = match_points(p1,TP,n1,TN,Btree);  % 
        stop2 = norm(T0-eye(4));                            % 
        if stop2 < 1e-5
            break;
        end
    end
    if abs(u1-u2)<1e-6                                      % 
        stop1 = 1;
    end
    count = count+i;                                                       % 
    u1 = u1/4;                                                             % 
    if u1<u2                                                               % 
        u1 = u2;
    end
    
end
p12 = T*[SP;ones(1,size(SP,2))]; 
p1 = p12(1:3,:); 
n1 = T(1:3,1:3)*SN;
[idx,dist] = knnsearch(Btree, p1');
inliers = dist<3*u2;                                                       % 
p1 = p1(:,inliers);
n1 = n1(:,inliers);
p2 = TP(:,idx(inliers));
n2 = TN(:,idx(inliers));

ww = weight(dist(inliers),3*u2)';
T0 = symm_po_pl(p1,p2,n1,n2,ww);
T = T0*T;



