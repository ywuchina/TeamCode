function T = symm_po_pl(p1,p2,n1,n2,w) 

w = w/sum(w);                         
w2 = sqrt(w);                         

meanp1 = mean(p1,2);                  
meanp2 = mean(p2,2);
p1 = p1-meanp1;
p2 = p2-meanp2;
n = n1+n2;                             
c = cross(p1,n);                       % (23) 
d = p2-p1;                             % (23) 

A = [c' n'];                           % (23)
b = dot(d,n)';                         % (23)

wm = repmat(w2',1,6);                  % (24)
A = A.*wm;
b = b.*w2';
AA = A'*A;                             % (24) 
Ab = A'*b;                             % (24) 

[U,S,V] = svd(AA);
S1 = inv(S);
x = U*S1*U'*Ab;                        % 

rot = x(1:3); trans = x(4:6);          % 
rotangle = norm(rot);                
TR = rotation_matrix(rotangle, rot);   % 
trans = trans + meanp2 - TR(1:3,1:3) * meanp1;                             % 
T = [TR(1:3,1:3) trans; 0 0 0 1];




