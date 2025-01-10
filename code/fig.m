clear all;clear; close all;clc;

tar = pcread('G:\featurefile\stanford\bunny\Bunny_045.ply');
src = pcread('G:\featurefile\stanford\bunny\Bunny_090.ply');

feat1 = extractFPFHFeatures(tar);
feat2 = extractFPFHFeatures(src);
[matchingPairs,scores] = pcmatchfeatures(feat1,feat2,tar,src,'Method','Approximate','MatchThreshold',0.1);
pts1 = src.Location(matchingPairs(:,2),:);
pts2 = tar.Location(matchingPairs(:,1),:); 

pt1 = pointCloud(pts2);
pt2 = pointCloud(pts1);

matran = load('G:\featurefile\stanford\bunny\tform.mat');
trtar = matran.tform{2};
trsrc = matran.tform{3};
new1 = pctransform(pt1, trtar);
new2 = pctransform(pt2, trsrc);
loc1 = new1.Location;
loc2 = new2.Location;

len = size(pts2,1);
num1 = 0;num2 = 0;
for i = 1:len
    dis(i) = norm(loc1(i,:)-loc2(i,:));
end
[num,index] = sort(dis);
matchedPts1 = select(tar,matchingPairs(index(1),1));
matchedPts2 = select(src,matchingPairs(index(1),2));
figure
pcshowMatchedFeatures(tar,src,matchedPts1,matchedPts2,"Method","montage");
% figure
% pcshowpair(new1,new2);