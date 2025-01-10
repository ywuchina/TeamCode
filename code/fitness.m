function error = fitness(x,task)

    inmodel = x>0.4;
    x(x>0.7) = 1;
    x(x<0.3) = 0;
    ind = x;
    ind(ind==0) = [];
    inmodel = find(x);
    feat_index = find(inmodel);
    feat1 = task.feat1(:,feat_index);
    feat2 = task.feat2(:,feat_index);
    feat1 = feat1.*ind;
    feat2 = feat2.*ind;
    [matchingPairs,~] = pcmatchfeatures(feat1,feat2,task.tarCloud,task.srcCloud,'Method','Approximate','MatchThreshold',0.03);
    matchedPts1 = select(task.tarCloud,matchingPairs(:,1));
    setpt = select(task.srcCloud,matchingPairs(:,2));
    pts2 = matchedPts1.Location;
    newl = pctransform(setpt, task.tform);
    newpt = newl.Location;

    len = size(pts2,1);
    num = 0;
    
    for t = 1:len
        if norm(newpt(t,:)-pts2(t,:)) < 0.3
            num = num + 1;
        end
    end
    error = 1 - (num / len);
end

