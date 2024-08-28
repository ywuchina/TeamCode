% 输入
% source_pc:        源点云，pointCloud类型
% gt_tform:         ground truth变换，rigid3d类型
% est_tform_cell:   算法估计的变换，cell类型，cell中的元素为rigid3d类型

% 输出
% 多幅error map图像，png格式
function [] = error_map(source_pc, gt_tform, est_tform_cell,dn,name)
    max_color_value = 3000;

    point_num = source_pc.Count;
    tform_num = length(est_tform_cell);

    mins = -inf(1, tform_num);
    maxs = zeros(1, tform_num);
    log_dist = zeros(point_num, tform_num);

    % 确定RMSE的最大最小对数值
    gt_pc = pctransform(source_pc, gt_tform);
    for i = 1:tform_num
        est_pc = pctransform(source_pc, est_tform_cell{i});

        dist = sqrt(sum((gt_pc.Location - est_pc.Location) .^ 2, 2));
        log_dist(:, i) = log10(dist);

        mins(1, i) = min(log_dist(:, i));
        maxs(1, i) = max(log_dist(:, i));
    end
    max_val = ceil(max(maxs) * 10) / 10;
    min_val = floor(min(mins) * 10) / 10;
    diff = max_val - min_val;

    for i = 1:tform_num
        % 对点云着色
        idx = max_color_value - ceil((max_val - log_dist(:, i)) ./ diff * max_color_value);
        idx(idx == 0) = 1;
        jet_color = colormap(jet(max_color_value)); 
        selected_color = jet_color(idx, :);
        gt_pc = pointCloud(gt_pc.Location, 'Color', selected_color);

        % 仅显示源点云，而且是经过ground truth变换后的源点云
        pcshow(gt_pc, 'BackgroundColor', [1 1 1]);hold on  
        view([0, 90]);
        axis off
        set(figure(1),'Position',[400,300,300,300]);
        ax=gca;
        ax.View=[89.474609375,57.3876953125];
        ax.CameraPosition=[1145.750010213174,-0.791190145056007,999.4712505680685];
        ax.CameraTarget=[-69.52527492621303,41.56052386928195,49.41730004318147];
        ax.CameraUpVector=[-0.610251573836611,-0.048389178138256,0.790728464182897];
        ax.CameraViewAngle=6.400198117782438;
        % 保存png图像操作
        set(gca,'LooseInset', get(gca,'TightInset'))
        frame = getframe(gca);
        im = frame2im(frame);
        imwrite(im, ['.\picture\reg\'+dn+'\'+name+'\'+num2str(i)+'.png']);

        close 1;
    end

    % 颜色条设置
    figure;
    clim([10^min_val, 10^max_val]);
    colorbar('Ticks',[10^min_val, 10^max_val],...
            'TickLabels',{['\fontsize{10} 10^{' num2str(min_val) '}'],...
            ['\fontsize{10} 10^{' num2str(max_val) '}']},...
            'Location', 'eastoutside',...
            'position',[0.5 0.2 0.07 0.7]);
    colormap(jet(max_color_value)); 
    set(figure(1),'Position',[400,300,100,200]);
    set(gcf,'color','white')
    axis off;
    set(gca,'LooseInset', get(gca,'TightInset'))
    frame = getframe(gca);
    im = frame2im(frame);
    imwrite(im, ['.\picture\reg\'+dn+'\'+name+'\'+num2str(i+1)+'.png']);
    close 1;
end