
function [p1,p2,n1,n2,r] = match_points(SP,TP,SN,TN,Btree)  

idx = knnsearch(Btree, SP');

p1 = SP;
n1 = SN;

p2 = TP(:,idx);
n2 = TN(:,idx);

r = sqrt(sum((p1-p2).^2));  
end
