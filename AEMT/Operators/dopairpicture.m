function dopairpicture(p1,p2,pair,T,yuzhi)
%成对点云作图
%
    l=[0x61,0xF4,0xE2]; r=[0xEC,0x6B,0xF7];
%     p1=dopicture(p1,l,r);
%     p2=dopicture(p2,l,r);
    p1=pointCloud(p1.Location,'Color',l);
    p2=pointCloud(p2.Location,'Color',r);
    np1=pointCloud(p1.Location(pair(:,1),:));
    np2=pointCloud(p2.Location(pair(:,2),:));
    tp1=pctransform(np1,affine3d(T));
    d=[];
    for q=1:length(tp1.Location(:,1))
        d(q)=rms(tp1.Location(q,:)-np2.Location(q,:));
    end
    fg=figure();
    pcshowMatchedFeatures(p1,p2,pointCloud(np1.Location(d<yuzhi,:)),pointCloud(np2.Location(d<yuzhi,:)));
    set(fg,'color','w');
end