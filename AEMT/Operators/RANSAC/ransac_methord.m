function [T, varargout] = ransac_methord( pts1,pts2,ransacCoef,funcFindTransf,funcDist)
    %[T, inlierIdx] = ransac1( pts1,pts2,ransacCoef,funcFindTransf,funcDist )
    %	Use Random Sample Consensus to find a fit from PTS1 to PTS2. 求从pts1到pts2的旋转参数 T
    %	PTS1 is M*n matrix including n points with dim M, PTS2 is N*n;  这里写错了，应该是NX3
    %	The transform, T, and the indices of inliers, are returned.
    %
    %	RANSACCOEF is a struct with following fields:   ransacCoef的参数解释如下
    %	minPtNum,iterNum,thDist,thInlrRatio
    %	MINPTNUM is the minimum number of points with whom can we 
    %	find a fit. For line fitting, it's 2. For homography, it's 4.
    %	ITERNUM is the number of iteration, THDIST is the inlier 
    %	distance threshold and ROUND(THINLRRATIO*n) is the inlier number threshold.
    %
    %	FUNCFINDF is a func handle, f1 = funcFindF(x1,y1)
    %	x1 is M*n1 and y1 is N*n1, n1 >= ransacCoef.minPtNum
    %	f1 can be of any type.
    %	FUNCDIST is a func handle, d = funcDist(f,x1,y1)
    %	It uses f returned by FUNCFINDF, and return the distance
    %	between f and the points, d is 1*n1.
    
    nout = max(nargout, 1);  % nargout 参数值根据返回值的个数确定，现在 nargout=2

    minPtNum = ransacCoef.minPtNum;       % 采样的点数
    iterNum = ransacCoef.iterNum;         % 迭代次数 
    thInlrRatio = ransacCoef.thInlrRatio; % 称为转换成功所需的内部匹配的百分比 
    thDist = ransacCoef.thDist;           % 判断是内点的阈值
    ptNum = size(pts1,1);                 % 原始点云的大小    
    thInlr = round(thInlrRatio*ptNum);    % 原始点云中用来做RANSA的点的数量【有点最低重叠率的意思,大于这个阈值表明现在的配准可能是正确的，再利用阈值内的这部分点，求个SVD细化】
    REFINE = ransacCoef.REFINE;           % 是否使用删选后的内点找到转换【有点再次寻优强化的意思】
    if isfield(ransacCoef, 'VERBOSE')     % ransacCoef中没设置变量VERBOSE，所以VERBOSE = 1
        VERBOSE = ransacCoef.VERBOSE;     % VERBOSE 控制是否输出信息,默认 1 输出
    else
        VERBOSE = 1;
    end

    inlrNum = zeros(1,iterNum);           %过程变量-迭代中某一次变换后在阈值内的点的数量 （1XiterNum double）
    inlrNum_refined = zeros(1,iterNum);   %过程变量-迭代中某一次变换后，再做一次SVD变换后内点的数量 （1XiterNum double）
    TForms = cell(1,iterNum);             %过程变量-代中某一次变换后的变换参数 （1XiterNum cell）

    for p = 1:iterNum 
        % 1. fit using  random points
        randomInts = randperm(ptNum);       % 产生ptNum长度的随机数列   
        sampleIdx = randomInts(1:minPtNum); % 根据数列在原始点云中随机取出三个点的序号
        f1 = funcFindTransf(pts1(sampleIdx, :),pts2(sampleIdx, :)); %根据这三个点，SVD求解变换参数 f1
        while(isempty(f1))
            randomInts = randperm(ptNum);       % 产生ptNum长度的随机数列   
            sampleIdx = randomInts(1:minPtNum); % 根据数列在原始点云中随机取出三个点的序号
            f1 = funcFindTransf(pts1(sampleIdx, :),pts2(sampleIdx, :)); %根据这三个点，SVD求解变换参数 f1
        end

        % 2. count the inliers, if more than thInlr, refit; else iterate计算内点数量，如果>thInlr,重新funcFindTransf，否则继续迭代
        dist = funcDist(f1,pts1,pts2); % 计算将pts1根据f1转换后与pts2的距离差 NX1
        inlier1 = find(dist < thDist); % 根据阈值，算一下在阈值内的点，inlier1是点的编号1，2，3，4.....
        inlrNum(p) = length(inlier1);  % 记录一下迭代中某一次变换后在阈值内的点的数量

        % refit if enough inliers, check that threshold is still met
        % 内点足够的话在此基础上重新进行funcFindTransf【相当于这部分所有点做一次SVD，跟之前的三个点做SVD不一样】，检查阈值是否仍然符合
        if length(inlier1) >= thInlr
            if REFINE
                f1_ref = funcFindTransf(pts1(inlier1, :),pts2(inlier1, :));
                if length(f1_ref)==0
                    continue;
                end
                dist = funcDist(f1_ref,pts1,pts2);
                inlier_refined = find(dist < thDist);
                inlrNum_refined(p) = length(inlier_refined);
                if inlrNum_refined(p) >= thInlr   %如果内点数量仍然足够的话，当前变换参数就是有效的，【更新】参数TForms
                    TForms{p} = f1_ref;
                end
            else
                TForms{p} = f1;
            end
        end      
    end

    % 3. choose the coef with the most inliers 找出最大内点数maxInliers 和 第几次idx
    if REFINE 
        [maxInliers,idx] = max(inlrNum_refined);
    else
        [maxInliers,idx] = max(inlrNum);
    end
    
    T = TForms{idx};  % 根据第几次是最大内点，找到变换矩阵T 
    FAILED = false;   % FAILED 当前的变换参数是否有问题
    try
        dist = funcDist(T,pts1,pts2);   % dist是离最近点的距离 【这还能不成立？？？？】
    catch
        % error('RANSAC could not find an appropriate transformation');
        if VERBOSE
            fprintf('RANSAC could not find an appropriate transformation\n');
        end
        FAILED = true;
        T = [];
        inlierIdx = [];
        numSuccess = 0;  % 成功的次数（内点数量>thInlr）
        maxInliers = 0;
    end
    
    if ~FAILED   % 当前的变换参数没有问题
        inlierIdx = find(dist < thDist); %判断是否是内点，inlierIdx 表示是内点的位置编号

        if REFINE
            numSuccess = sum(inlrNum_refined >= thInlr); % 成功的次数（内点数量>thInlr）
        else
            numSuccess = sum(inlrNum >= thInlr);         % 成功的次数（内点数量>thInlr）
        end
        if VERBOSE
            fprintf('RANSAC succeeded %d times with a maximum of %d Inliers (%0.2f %%)\n', numSuccess, maxInliers, 100*maxInliers/ptNum);
        end
    end
	
    % outputs        
    if nout > 1
        varargout{1} = inlierIdx;
    end
    if nout > 2
        varargout{2} = numSuccess;
    end 
    if nout > 3
        varargout{3} = maxInliers;
    end 
    if nout > 4
        varargout{4} = 100*maxInliers/ptNum;
    end 
    
end
