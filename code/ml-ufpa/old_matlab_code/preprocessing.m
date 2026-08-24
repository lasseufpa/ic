x1=[3 -4 50]
x2=[40 40 -6]
x3=[-5 4 500]
x=[3 40 -5]
% Normalization
min_x = min(x)
span_x = max(x) - min_x

y=(x-min_x)/span_x
mean(y)
std(y)

% Standardization
average_x = mean(x)
standard_deviation_x = std(x)

y=(x-average_x)/standard_deviation_x

mean(y)
std(y)