function d = dominates(cost_1, cost_2)
%     if cost_1(1) + 0.2 <= cost_2(1)
%         d = true;
%     else
        d = all(cost_1 <= cost_2) && any(cost_1 < cost_2);
%     end
end
