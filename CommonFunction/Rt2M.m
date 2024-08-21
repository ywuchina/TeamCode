
function M= Rt2M(R,t)   

M(1:3,1:3)= R;
M(4,4)= 1;
M(1:3,4)= t;