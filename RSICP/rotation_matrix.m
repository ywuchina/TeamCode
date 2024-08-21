function R = rotation_matrix(angle, rot)

rx = rot(1); ry = rot(2); rz = rot(3);
l = sqrt(rx*rx + ry*ry + rz*rz);
if l==0
    R = eye(4);
end
l1 = 1/l; x = rx * l1; y = ry * l1; z = rz * l1;
s = sin(angle); c = cos(angle);
xs = x*s; ys = y*s; zs = z*s; c1 = 1 - c;
xx = c1*x*x; yy = c1*y*y; zz = c1*z*z;
xy = c1*x*y; xz = c1*x*z; yz = c1*y*z;
R = [xx+c,  xy+zs, xz-ys, 0,
				xy-zs, yy+c,  yz+xs, 0,
				xz+ys, yz-xs, zz+c,  0,
				0, 0, 0, 1]';
end