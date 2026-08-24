%% Dataset "simple":
%12,3.2,0
%10,0.5,1
%14,2.8,0
%14,2.4,0
%13,1.8,1
%13.8,1.5,0
%11,1,1

%% I) Bayes classifier using only the first dimension x1
%Class 0:
prior0=4/7
m0l=mean([12 14 14 13.8])    %13.450
s0l=std([12 14 14 13.8])         %0.97125
%Class 1:
prior1=3/7
m1l=mean([10 13 11])           %11.333
s1l=std([10 13 11])                %1.5275
%Test example: 
x=14.3;
num0=normpdf(x,m0l,s0l)*prior0
num1=normpdf(x,m1l,s1l)*prior1
Px=num0+num1; %denominator, just a scale factor
posterior0=num0/Px  %0.90409
posterior1=num1/Px  %0.095912

%% II) Naive Bayes using two dimensions
%Class 0:
prior0=4/7
m0l=mean([12 14 14 13.8])    %13.450
m0w=mean([3.2 2.8 2.4 1.5]) %2.4750
s0l=std([12 14 14 13.8])         %0.97125
s0w=std([3.2 2.8 2.4 1.5])      %0.72744
%Class 1:
prior1=3/7
m1l=mean([10 13 11])           %11.333
m1w=mean([0.5 1.8 1])         %1.1000
s1l=std([10 13 11])                %1.5275
s1w=std([0.5 1.8 1])              %0.65574
%Test example: 
x=[13 3.2]; xl=x(1); xw=x(2); 
num0=normpdf(xl,m0l,s0l)*normpdf(xw,m0w,s0w)*prior0
num1=normpdf(xl,m1l,s1l)*normpdf(xw,m1w,s1w)*prior1
Px=num0+num1; %denominator, just a scale factor
posterior0=num0/Px  %0.99685
posterior1=num1/Px  %0.0031541
