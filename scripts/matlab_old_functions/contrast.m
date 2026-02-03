function [contrast] = contrast(ref)

[m,n] = size(ref);
hist = imhist(ref);
p = hist./(m*n);
mean_var = mean(p);
i = (1:length(p))';

suma = p.*(i-mean_var).^2;
contrast = sqrt(sum(suma));