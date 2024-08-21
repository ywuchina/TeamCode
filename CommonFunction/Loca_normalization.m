
function [norm_pc, T] = Loca_normalization(pc)
    location = zeros(size(pc.Location));
    T = zeros(1,3);
    for i = 1:3 
       temp = mean(pc.Location(:, i));
       T(i) = temp;
       temp = temp * ones(size(pc.Location, 1), 1);
       location(:, i) = pc.Location(:, i) - temp;
    end
    norm_pc = pointCloud(location); 
end