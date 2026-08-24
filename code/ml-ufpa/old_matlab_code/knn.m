%conjunto de treino
X=[  -4000      4      2      0
   2000      2      4      0
  -4000     -6     -3      1
  -6000     -1     -4      1
      0      0     -3      1];
%vetor de teste:
xtest = [-5000 2 4];
ytest = 1;
%KNNs individuais com K=3 – 
%Assuma o seu (matricula) conjunto de treino.
%usar o 1º vetor do conjunto de teste (não usar validação)
%Responder: indicando a menor distância e a classe predita quando K=3
Ntrain = 5 %numero de exemplos no conjunto de treino
min_distancia = Inf; %inicializacao da menor distancia
for i=1:Ntrain %busca pelo vizinho mais proximo
  xtrain = X(i,1:3); %extrai o i-esimo vetor de entrada
  ytrain = X(i,4); %extrai a class correta do i-esimo exemplo
  distancia = norm(xtrain-xtest) %distancia Euclideana
  if distancia  < min_distancia %verifica se deve atualizar
    min_distancia = distancia; %atualiza
    vizinho_mais_proximo = i;
  endif
end
min_distancia 
vizinho_mais_proximo 
classe_predita = X(vizinho_mais_proximo,4)
if classe_predita == ytest
  disp('Acertou')
else
  disp('Errou')
endif
