function [] = logger(message)
    disp([datestr(now), ' - ', message]);
end
