N =1000; thresholds = linspace (-3.5,5.5,N); Pe = zeros (1,N);
for i=1:N % loop over the defined grid of thresholds
  Pe0 =1/3*(5-thresholds(i)); %Pe0 is the conditional P(e/y=0)
  Pe1 =1/2*( thresholds(i)-4);%Pe1 is the conditional P(e/y=1)
  if Pe0 <0 Pe0 =0; end %a probability cannot be negative
  if Pe1 <0 Pe1 =0; end %a probability cannot be negative
    Pe(i) = 0.4* Pe0 +0.6* Pe1 ;% prob . error for thresholds (i)
  end
plot ( thresholds , Pe); % visualize prob . for each threshold
xlabel('threshold')
ylabel('P(e)')