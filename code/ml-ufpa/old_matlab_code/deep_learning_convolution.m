%entrada
X=[1 0 3 0
250 2 0 0
3 1 255 1
1 0 3 2];

%entrada com padding
X=[1 0 3 0 0
250 2 0 0 0
3 1 255 1 0
1 0 3 2 0
0 0 0 0 0];

%filtros
F111=[1 -1
1 -1];

F121=[1 1
-1 -1];

F211=[2 -1
1 0];

F212=[-2 1
0 -1];

%% 1a opcao, usando xcorr2
%1a camada
Y1 = xcorr2(X,F111);
Y2 = xcorr2(X,F121);

%exclude first and last row and column because xcorr2 does not use
%only the "valid" regions
Y1=Y1(2:end-1,2:end-1)
Y2=Y2(2:end-1,2:end-1)

H1 = max(Y1,0)
H2 = max(Y2,0)

P1 = xcorr2(H1,F211)
P2 = xcorr2(H2,F212)

%jump first row and column and mimic stride=2
P1=P1(2:2:end,2:2:end)
P2=P2(2:2:end,2:2:end)

P=P1+P2
H3 = max(P,0)

%% 2a opcao, usando conv2
%lembre que a convolucao "rebate" sinais e correlacao nao
%em deep learning, o povo chama de camada convolucional, mas na verdade
%implementa correlacao, sem haver rebatimento. Daí temos que usar o truque
%de rebater ao se querer usar conv2:

%A conv2 com opcao 'same' vai tomar conta do padding. Daí não se precisa
%usa-lo. Basta usar o X sem padding:
X=[1 0 3 0
250 2 0 0
3 1 255 1
1 0 3 2];

Y1c = conv2(X,fliplr(flipud(F111)),'same')
Y2c = conv2(X,fliplr(flipud(F121)),'same')

H1c = max(Y1c,0)
H2c = max(Y2c,0)

P1c = conv2(H1c,fliplr(flipud(F211)),'same')
P2c = conv2(H2c,fliplr(flipud(F211)),'same')
Pc=P1c+P2c
H3c = max(Pc,0)
H3c = H3c(1:2:end,1:2:end)
