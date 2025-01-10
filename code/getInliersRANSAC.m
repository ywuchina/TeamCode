function [tform] = getInliersRANSAC(srcCloud,tarCloud,R,feat1,feat2)
% select loc1M / loc1S to match the query with the correct crop         选择 loc1M / loc1S 将查询与正确的裁剪相匹配
% select loc2M / loc2S to match the query with the wrong crop. Ideally, 选择 loc2M / loc2S 以匹配错误作物的查询。 
% RANSAC should fail in this case and not return a transformation       理想情况下，RANSAC 在这种情况下应该失败并且不返回转换


%  gridStep = 0.3;                % 采样步长
%  threshold = gridStep*gridStep;      % RANSAC中的参数
% % srcCloudDown = pcdownsample(srcCloud, 'gridAverage', gridStep); % 1x1 pointCloud 3XN 388
% % tarCloudDown = pcdownsample(tarCloud, 'gridAverage', gridStep); % 1x1 pointCloud 3XN 352
% srcCloudDown = pcFarthestPointSample(srcCloud,ceil(srcCloud.Count*0.025));
% tarCloudDown = pcFarthestPointSample(tarCloud,ceil(tarCloud.Count*0.025));
% % 提取特征(每个点都提取)
% [fixedFeature] = extractFPFHFeatures(tarCloudDown);
% [movingFeature] = extractFPFHFeatures(srcCloudDown);
srcCloudDown = srcCloud;
tarCloudDown = tarCloud;

%0-1 subf = [1,2,4,6,7,8,9,11,12,14,15,18,19,21,22,23,24,25,26,27,28,29,30,31,33,34,35,36,38,40];
%1-2 subf = [1,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,22,23,25,26,27,28,29,30,31,32,33,34,35,36,37,38,40];
%0-3 subf = [2,3,4,6,8,10,11,12,14,16,17,19,20,21,22,26,27,28,29,30,31,32,33,35,37,39,40];
%1-3 subf = [1,2,3,4,5,7,8,9,10,11,14,15,17,18,19,21,23,25,26,28,29,30,31,32,33,35,36,38,39];
% kids subf = [1,2,4,5,8,9,10,11,12,14,16,17,18,19,21,22,23,25,26,27,28,29,30,31,32,33,34,36,38,39,40];
% subf = [2,4,5,6,7,8,9,11,12,15,16,17,18,21,22,23,25,26,27,31,32,34,35,36,37,39,40];
% feat1 = feat1(:,subf);
% feat2 = feat2(:,subf);

fixedFeature = feat2;
movingFeature = feat1;
% 特征匹配的点
[matchingPairs,scores] = pcmatchfeatures(fixedFeature,movingFeature,tarCloudDown,srcCloudDown,'Method','Approximate','MatchThreshold',0.03);   % Fast Global Registration论文的方法

% 可视化特征匹配结果
matchedPts1 = select(tarCloudDown,matchingPairs(:,1));
matchedPts2 = select(srcCloudDown,matchingPairs(:,2));
pcshowMatchedFeatures(tarCloudDown,srcCloudDown,matchedPts1,matchedPts2,"Method","montage");
title('特征匹配结果');

% 根据匹配，得到相应的坐标
matchfixedCloud = pointCloud(tarCloudDown.Location(matchingPairs(:,1),:));  
matchmovingFeature = pointCloud(srcCloudDown.Location(matchingPairs(:,2),:));

srcSeed = matchmovingFeature.Location; % N X3 
tarSeed = matchfixedCloud.Location;    % N X3  

pts1 = srcSeed; % data    NX3 
pts2 = tarSeed; % Model
%% 测试对应点准确率
% 转pts2
% setpt = pointCloud(pts2);
% newl = pctransform(setpt, R);
% newpt = newl.Location;
% % 求newpt 与 pts1的欧氏距离
% len = size(pts1,1);
% for i = 1:len
%     dis(i) = norm(newpt(i,:)-pts1(i,:));
% end
% [num1,index] = sort(dis);
%% 提取部分重叠点云
% 获得种子点
% len = size(pts1,1);
% dist = zeros(len,len);  %存储了
% men = zeros(len,1);
% for i = 1:len
%     for j = i+1:len
%         dist(i,j) = abs(norm(pts1(i,:)-pts1(j,:)) - norm(pts2(i,:)-pts2(j,:)));
%         dist(j,i) = dist(i,j);
%     end
%     for k = 1:len
%         if dist(i,k) < 0.01
%             men(i) = men(i) + 1;
%         end
%     end
%     %men(i) = mean(dist(i,:));  均值法
% end
% [num2,index] = sort(men,'DESCEND');  % index为升序排列的对应点的索引
% % 根据对应点聚类形成两个簇 pt1,pt2
% num_root = 150;  %种子点个数
% clu = pts1(index(1:num_root),:);
% idx = kmeans(clu,2);
% nu1=1;
% nu2=1;
% for i = 1:num_root
%     if idx(i) == 1
%         pt1(nu1) = index(i);
%         nu1 = nu1+1;
%     else
%         pt2(nu2) = index(i);
%         nu2 = nu2+1;
%     end
% end
% % 根据索引找出对应点坐标，然后聚类生成局部点云
% trpa=[];trpb=[];trpc=[];trpd=[];
% for i=1:size(pt1,2)
%     [trp1,~] = findNeighborsInRadius(srcCloud,pts1(pt1(i),:),0.008);
%     [trp2,~] = findNeighborsInRadius(tarCloud,pts2(pt1(i),:),0.008);
%     trpa = [trpa',trp1']';
%     trpb = [trpb',trp2']';
% end
% for i=1:size(pt2,2)
%     [trp3,~] = findNeighborsInRadius(srcCloud,pts1(pt2(i),:),0.008);
%     [trp4,~] = findNeighborsInRadius(tarCloud,pts2(pt2(i),:),0.008);
%     trpc = [trpc',trp3']';
%     trpd = [trpd',trp4']';
% end
% trpa=unique(trpa);trpb=unique(trpb);trpc=unique(trpc);trpd=unique(trpd);
% % 可视化局部点云
% figure
% pcshow(srcCloud)
% hold on
% %plot3(pts1(index(1),1),pts1(index(1),2),pts1(index(1),3),'*r')
% plot3(srcCloud.Location(trpa,1),srcCloud.Location(trpa,2),srcCloud.Location(trpa,3),'*r')
% plot3(srcCloud.Location(trpc,1),srcCloud.Location(trpc,2),srcCloud.Location(trpc,3),'*b')
% figure
% pcshow(tarCloud)
% hold on
% %plot3(pts2(index(1),1),pts2(index(1),2),pts2(index(1),3),'*r')
% plot3(tarCloud.Location(trpb,1),tarCloud.Location(trpb,2),tarCloud.Location(trpb,3),'*r')
% plot3(tarCloud.Location(trpd,1),tarCloud.Location(trpd,2),tarCloud.Location(trpd,3),'*b')
% set1 = pointCloud(srcCloud.Location(trpa,:));
% set2 = pointCloud(tarCloud.Location(trpb,:));
% set3 = pointCloud(srcCloud.Location(trpc,:));
% set4 = pointCloud(tarCloud.Location(trpd,:));
% pcwrite(set1,'trp1.ply');
% pcwrite(set2,'trp2.ply');
% pcwrite(set3,'trp3.ply');
% pcwrite(set4,'trp4.ply');
%% RANSAC 配准
% set RANSAC Parameters
% number of sampled points. minimum is 3 for rigid body transformation  每次采样的点数
coeff.minPtNum = 3; 

% number of iterations                                         % RANSAC 迭代次数（多少次随机取点）
coeff.iterNum = 3e3; 

% distance (in world units世界单位) below which matches are considered inliers  判断是内点的阈值
%coeff.thDist = 5*threshold; 
coeff.thDist = 0.01;
% percentage of matches that are inliers needed to call the transformation a success 称为转换成功所需的内部匹配的百分比，最后用来计算一个sucess rate
coeff.thInlrRatio = 0.1; 

%% Perform RANSAC with rigid transform T and distance function
% REFINE: find the transformation again using all inliers, if successful 是否继续使用筛选后的内点再找一次转换
coeff.REFINE = true;

tic    %pts1(NX3),pts2(NX3),coeff(1x1 struct[minPtNum,iterNum,thDist,thInlrRatio,REFINE]),@estimateTransform句柄函数，@calcDists句柄函数
[T, inlierPtIdx] = ransac_methord(pts1,pts2,coeff,@estimateTransform,@calcDists);
toc

%% Use returend T to align pts1 (Model) with pts2 (Surface)
% if ~isempty(T)
%     pts1_aligned = [pts1, ones(size(pts1, 1), 1)] * T;
%     pts1_aligned = pts1_aligned(:, 1:3);
% end

%% pts1_aligned 是配准后的点集  T 是tform类型的
   tform = T ;
end