% 输入： pts1 NX3  pts1 NX3
% 输出： T [pts1, 1] * T = [pts2, 1]
%% function that estimates the 3D rigid transformation from at least 3 pts  SVD方法
function T = estimateTransform(pts1, pts2)
% this function estimates the rigid transformation T between pts1 and pts2
% based on SVD. At least three points are needed for the 3D transformation. 

% returns T, such that [pts1, 1] * T = [pts2, 1]

    num_points = size(pts1, 1);
    
    % make sure that the rank full
    if rank(pts1) < 3 || rank(pts2) < 2
        T = [];
        return;
    end
    
    % if points only span a plane, add a 4th point to make a full rank
    % decomposition (normal vector) 如果点仅跨越一个平面，则添加第 4 个点以进行全秩分解（法向向量）
    if num_points == 3     
        % use median as start for normal vector 使用中值作为法向量的起点
        c1_L1 = mean(pts1);
        c2_L1 = mean(pts2);
         
        % use cross product as direction for normal vector  使用叉积作为法向量的方向
        n1 = cross(pts1(3, :) - pts1(2, :), pts1(3, :)  - pts1(1, :), 2);
        n2 = cross(pts2(3, :) - pts2(2, :), pts2(3, :)  - pts2(1, :), 2);
        
        % use median distance of points as length for normal vector 使用点的中值距离作为法向量的长度
        l1 = median(vecnorm(pts1-circshift(pts1, 1, 1), 2, 2));
        l2 = median(vecnorm(pts2-circshift(pts2, 1, 1), 2, 2));
        
        % get additional point 获得额外点，就为了增加一个点？？？？？？？？
        p1 = c1_L1 + (n1/norm(n1))*l1;
        p2 = c2_L1 + (n2/norm(n2))*l2;
        
        pts1 = [pts1; p1];
        pts2 = [pts2; p2];
    end
    
    %% 下面是常规的SVD求解变换矩阵
    % transform to dimensions used in paper so that each point is a column vector 转换维度
    d = pts1'; % N x 3 --> 3 x N
    m = pts2'; % N x 3 --> 3 x N
    
    % implemented following the paper [eggert_comparison_mva97]
    % get centroid for each point set and center around (0,0)
    cd = mean(d, 2);
    cm = mean(m, 2);
    
    %DEBUG
    if sum(isnan(cd)) + sum(isnan(cm)) > 0
        pause();
    end
    
    % centered points 去中心化
    d_c = d - cd;
    m_c = m - cm;
    
    H = m_c*d_c';
       
    [U, S, V] = svd(H); % H = U*S*V'
    
    R = V*U';
    t = cd - R*cm;
    
    % now build transform matrix, so that [d; 1] = TF * [m; 1]    
    TF = eye(4);
    TF(1:3, 1:3) = R;
    TF(1:3, 4) = t;
    
    % return matrix, so that [pts1, 1] * T = [pts2, 1]
    T = TF';


end