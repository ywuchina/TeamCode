function VisualizeFun(optModel, params)
% Mov0 = params.Mov0; 
Ref0 = params.Ref0; 
Aft = params.Aft;
P = cat(1, optModel(:).prior);
%%%%%%%%%%
%% plot registration results.
nDim = size(Ref0, 1);
figure;
hold on;
grid on;
% axis equal;
if nDim == 3
    set(gcf,'Position',[0 0 800 600], 'color', 'w');
    set(gca,'Position',[0.01 0.01 0.99,0.99]);
end
Str = {};
h = [];
if nDim == 3
    hold on;
    h = [h plot3(Ref0(1, :), Ref0(2, :), Ref0(3, :), 'g.')];
    Str{end+1} = 'target';
else
    plot(Ref0(1, :), Ref0(2, :), 'g.');
end
Z = params.Z;
Sita = cat(1, optModel(:).para); 
if Sita(1) > Sita(2)
    optModel = optModel([2 1]); 
    Z        = Z([2 1], :); 
end
if size(Z, 1) > 1
    [~, Label] = max(Z);
else
    Label = Z;
end
P = cat(2, optModel(:).prior)
Coeff = cat(2, optModel(:).coeff)
C = unique(Label);
ColorArray = {'r', 'b'};
hhh = []; 
if params.save
    filename = fullfile(pwd, 'cloud_ref.pcd');
    pcwrite(pointCloud(Ref0'), filename);
end
for id = 1 : 1 : length(C)
    selId = C(id);
    idx = find(Label == selId);
    colorId = mod(C(id), length(ColorArray));
    if colorId == 0
        colorId = length(ColorArray);
    end
    data = Aft(:, idx);
    if params.save
        if id == 1
            filename = fullfile(pwd, 'cloud_outlier.pcd'); 
        else
            filename = fullfile(pwd, 'cloud_inlier.pcd'); 
        end
        pcwrite(pointCloud(data'), filename); 
    end
    color = ColorArray{colorId};
    if nDim == 2
        ax = plot(data(1, :), data(2, :), 'color', color, 'marker', 'o', 'markersize', 3, 'linestyle', 'none' );
    else
        hhh(end+1) = pcshow(data', color, 'markersize', 50);
    end
%     str = sprintf('source with $n_i$ = %.2f,$\\theta$ = %.2f', ...
%         optModel(selId).prior, optModel(selId).para );
    if id == 1
        str = 'outlier point cloud'; 
    else
        str = 'inlier  point cloud'; 
    end
    str = sprintf('%s, $${\\rm E}[\\tau_%d] = %.2f$$', str, id, optModel(selId).para );
    Str{end+1} = str;
    hold on;
end
legend(Str, 'box', 'off', 'location', 'west', 'FontSize', 22, ...
    'FontWeight', 'bold', 'interpreter', 'latex');
xlabel('X/m', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Y/m', 'FontSize', 16, 'FontWeight', 'bold');
zlabel('Z/m', 'FontSize', 16, 'FontWeight', 'bold');
title('Outlier Awareness', 'FontSize', 16, 'FontWeight', 'bold');
end