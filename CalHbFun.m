function [H, b] = CalHbFun(params, bFast)
Aft = params.Aft;
w   = params.W;
MArray = params.MArray;
V = params.V;
if bFast
    %%%%%% assign to variables.
    m1 = MArray(1,:);
    m2 = MArray(2,:);
    m3 = MArray(3,:);
    m4 = MArray(4,:);
    m5 = MArray(5,:);
    m6 = MArray(6,:);
    v1 = V(1, :);
    v2 = V(2, :);
    v3 = V(3, :);
    p1 = Aft(1, :); 
    p2 = Aft(2, :); 
    p3 = Aft(3, :); 
    %%%%%%% Hrr.
    Hrr = [];
    Hrr(end+1, :) = w.*(m4.*(p3.*p3) + m6.*(p2.*p2) - m5.*p2.*p3*2.0);
    Hrr(end+1, :) = -p3.*(m2.*p3.*w - m3.*p2.*w) + p1.*(m5.*p3.*w - m6.*p2.*w);
    Hrr(end+1, :) = p2.*(m2.*p3.*w - m3.*p2.*w) - p1.*(m4.*p3.*w - m5.*p2.*w);
    Hrr(end+1, :) = -p3.*(m2.*p3.*w - m5.*p1.*w) + p2.*(m3.*p3.*w - m6.*p1.*w);
    Hrr(end+1, :) = w.*(m1.*(p3.*p3) + m6.*(p1.*p1) - m3.*p1.*p3*2.0);
    Hrr(end+1, :) = -p2.*(m1.*p3.*w - m3.*p1.*w) + p1.*(m2.*p3.*w - m5.*p1.*w);
    Hrr(end+1, :) = p3.*(m2.*p2.*w - m4.*p1.*w) - p2.*(m3.*p2.*w - m5.*p1.*w);
    Hrr(end+1, :) = -p3.*(m1.*p2.*w - m2.*p1.*w) + p1.*(m3.*p2.*w - m5.*p1.*w);
    Hrr(end+1, :) = w.*(m1.*(p2.*p2) + m4.*(p1.*p1) - m2.*p1.*p2*2.0);
    Hrr = reshape( sum(Hrr, 2), 3, 3)';
    %%%% Hrt.
    Hrt = [];
    Hrt(end+1, :) = -w.*(m2.*p3 - m3.*p2);
    Hrt(end+1, :) = -w.*(m4.*p3 - m5.*p2);
    Hrt(end+1, :) = -w.*(m5.*p3 - m6.*p2);
    Hrt(end+1, :) = w.*(m1.*p3 - m3.*p1);
    Hrt(end+1, :) = w.*(m2.*p3 - m5.*p1);
    Hrt(end+1, :) = w.*(m3.*p3 - m6.*p1);
    Hrt(end+1, :) = -w.*(m1.*p2 - m2.*p1);
    Hrt(end+1, :) = -w.*(m2.*p2 - m4.*p1);
    Hrt(end+1, :) = -w.*(m3.*p2 - m5.*p1);
    Hrt = reshape( sum(Hrt, 2), 3, 3)';
    %%%% Htt.
    Htt = [];
    Htt(end+1, :) = m1.*w;
    Htt(end+1, :) = m2.*w;
    Htt(end+1, :) = m3.*w;
    Htt(end+1, :) = m2.*w;
    Htt(end+1, :) = m4.*w;
    Htt(end+1, :) = m5.*w;
    Htt(end+1, :) = m3.*w;
    Htt(end+1, :) = m5.*w;
    Htt(end+1, :) = m6.*w;
    Htt = reshape( sum(Htt, 2), 3, 3)';
    %%%%% reduce to H.
    H = [Hrr  Hrt
        Hrt' Htt];
    %%%%%% br
    br = [];
    br(end+1, :) = -v1.*(m2.*p3.*w - m3.*p2.*w) - v2.*(m4.*p3.*w - m5.*p2.*w) - v3.*(m5.*p3.*w - m6.*p2.*w);
    br(end+1, :) = v1.*(m1.*p3.*w - m3.*p1.*w) + v2.*(m2.*p3.*w - m5.*p1.*w) + v3.*(m3.*p3.*w - m6.*p1.*w);
    br(end+1, :) = -v1.*(m1.*p2.*w - m2.*p1.*w) - v2.*(m2.*p2.*w - m4.*p1.*w) - v3.*(m3.*p2.*w - m5.*p1.*w);
    br = sum(br, 2);
    %%%%%% bt
    bt = []; 
    bt(end+1, :) = w.*(m1.*v1 + m2.*v2 + m3.*v3);
    bt(end+1, :) = w.*(m2.*v1 + m4.*v2 + m5.*v3);
    bt(end+1, :) = w.*(m3.*v1 + m5.*v2 + m6.*v3);
    bt = sum(bt, 2);
    %%%%%% reduce to b.
    b = [br; bt];
else
    H = zeros(6, 6);
    b = zeros(6, 1);
    parfor id = 1 : 1 : size(V, 2)
        v = V(:, id);
        pt_aft = Aft(:, id);
        A = [-SkewFun(pt_aft) eye(3)];
        w = W(id);
        m = MArray(:, id);
        M = [m(1) m(2) m(3)
            m(2) m(4) m(5)
            m(3) m(5) m(6)];
        H = H + w * A'*M*A;
        b = b + w * A'*M*v;
    end
end
end

