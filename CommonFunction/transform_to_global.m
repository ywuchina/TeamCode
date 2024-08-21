function p = transform_to_global(p, R, t)
% rotate
p(1:3,:) = R*p(1:3,:);
% translate
p(1,:) = p(1,:) + t(1);
p(2,:) = p(2,:) + t(2);
p(3,:) = p(3,:) + t(3);