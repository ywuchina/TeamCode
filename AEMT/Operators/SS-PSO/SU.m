% %������������֮��Ļ���Ϣ
% %u1��������������1
% %u2��������������2
% %wind_size�������ĳ���
function SU = SU(u1, u2)

wind_size=size(u1,1);
x = [u1, u2];
n = wind_size;
[xrow, xcol] = size(x);
bin = zeros(xrow,xcol);
pmf = zeros(n, 2);
for i = 1:2
    minx = min(x(:,i));
    maxx = max(x(:,i));
    binwidth = (maxx - minx) / n;
    edges = minx + binwidth*(0:n);
    histcEdges = [-Inf edges(2:end-1) Inf];
    [occur,bin(:,i)] = histc(x(:,i),histcEdges,1); %ͨ��ֱ��ͼ��ʽ���㵥��������ֱ��ͼ�ֲ�
    pmf(:,i) = occur(1:n)./xrow;
end
%����u1��u2�����ϸ����ܶ�
jointOccur = accumarray(bin,1,[n,n]);  %��xi��yi����������ͬʱ����n*n�ȷַ����е�������Ϊ���ϸ����ܶ�
jointPmf = jointOccur./xrow;
Hx = -(pmf(:,1))'*log2(pmf(:,1)+eps);
Hy = -(pmf(:,2))'*log2(pmf(:,2)+eps);
Hxy = -(jointPmf(:))'*log2(jointPmf(:)+eps);
MI = Hx+Hy-Hxy;%����Ϣ
mi = MI/sqrt(Hx*Hy);
% ���ߣ�Reacubeth 
% ��Դ��CSDN 
% ԭ�ģ�https://blog.csdn.net/xyisv/article/details/81745752 
% ��Ȩ����������Ϊ����ԭ�����£�ת���븽�ϲ������ӣ�
SU=2*(Hx+Hy-Hxy)/(Hx+Hy);%�ԳƲ�ȷ����

