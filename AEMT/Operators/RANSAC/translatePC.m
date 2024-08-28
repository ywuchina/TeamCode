function np=translatePC(p,v)
%平移点云
%   p是点云 v是移动的方向[x,y,z]
    nl=p.Location+repmat(v,size(p.Location,1),1);
end