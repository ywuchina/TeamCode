function [error_t, error_r] = cal_error(tform, gt_tform)
    error_t = norm(gt_tform.Translation - tform.Translation);
    
    if (trace(gt_tform.Rotation' * tform.Rotation) - 1) / 2 <= 1
        error_r = rad2deg( acos( (trace(gt_tform.Rotation' * tform.Rotation) - 1) / 2 ) );
    else
        fprintf("overstep the domain of arccosin\n");
        temp = 1 - ((trace(gt_tform.Rotation' * tform.Rotation) - 1) / 2 - 1);
        error_r = rad2deg(acos(temp));
    end
%     error_r = norm(gt_tform.Rotation' * tform.Rotation - eye(3));
end