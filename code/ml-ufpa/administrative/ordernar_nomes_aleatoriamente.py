#%%
# Modificar na lista
#By Lorhan Rodrigues
import random

lista = ['Huguinho', 'Zezinho', 'Luisinho']
novalista = []

# A "novalista" é gerada utilizando a "lista", o primeiro parâmetro passado é a lista original, o segundo parâmetro passado é o tamanho da lista, len vem de length, esse valor que len(lista) gera é um inteiro.
novalista = random.sample(lista, len(lista))
# Print da nova lista
print(novalista)

#-----------------------------------

#%%
# Inserir nome por nome
import random

inserir = True
lista = []
novalista = []

while inserir:
    # Inserindo nome na lista
    nome = (input('Inserir nome ou "P" para parar.'))

    # "P" irá parar a adição de itens
    if nome == 'p' or nome == 'P':
        inserir = False
    else:
        lista.append(nome)

# Reordenando nomes aleatoriamente
novalista = random.sample(lista, len(lista))
# Print da nova lista
print(novalista)

#-----------------------------------

# %%
# Inserir lista separada por vírgula. Exemplo: "Huguinho, Zezinho, Luisinho"
import random

inserir = True
lista = []
novalista = []

while inserir:
    # Inserindo nome na lista
    lista = (input('Inserir lista de nomes separados por vírgula.'))
    # Removendo espaços
    lista = lista.replace(" ", "")
    # Separando string em uma lista
    lista = lista.split(",")
    inserir = False

# Reordenando nomes aleatoriamente
novalista = random.sample(lista, len(lista))
# Print da nova lista
print(novalista)

#-----------------------------------

#%%
