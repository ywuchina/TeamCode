
function X=data_t(fly)

if fly==1
    load('Ovarian.mat')
    [kk,k1]=size(Data);
    for i=1:k1
    max1=max(Data(:,i));
    min1=min(Data(:,i));
    Data(:,i)=(Data(:,i)-min1)/(max1-min1);
    end 
    X=Data;
    Y=label;
    for jj=1:kk
        if Y(jj,1)==0
            Y(jj,1)=1;
        elseif Y(jj,1)==1
            Y(jj,1)=2;
        end
    end
    X=[Y,X]; 
end
    
    
