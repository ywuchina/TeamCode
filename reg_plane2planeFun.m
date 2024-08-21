function [dR, dT] = reg_plane2planeFun(params)
[H, b] = CalHbFun(params, 1);
%%%%%%%%%%% solve the problem || Ax - b ||, the solution also can be x = pinv( A' * A) * A' * b - x
dx = -pinv(H)*b;
dR = expm(SkewFun(dx(1:3))); % R0'*rstR;
dT = dx(4:6); % (rstT - T0);
end