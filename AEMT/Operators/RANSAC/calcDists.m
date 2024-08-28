%% function that calculates the distance between points after transform T
% 输入 pts2经过T的变换后，与pts1的距离和
% 输出 d(NX1)
function d = calcDists(T,pts1,pts2) %这里应该是不对的，[pts1, 1] * T = [pts2, 1]，为什么代码是pts2*T？？？
% Project PTS2 to PTS2_trans using the rigid transform T, then calcultate the distances between
% PTS1 and PTS2_trans

    pts2(:, 4) = 1;
    pts2_trans = pts2 * T;
    pts2_trans = pts2_trans(:, 1:3);
    d = sum((pts1-pts2_trans).^2,2); %这里因为点是一一对应的，所以直接相减即可，按理说应该建树，求最近点
    
%     kdtree = KDTreeSearcher(pts1);
%     [~, dist] = knnsearch(kdtree, pts2_trans);
%     d = sum(dist);


end