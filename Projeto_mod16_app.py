import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_text, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import graphviz

sns.set(context='talk', style='ticks')

st.set_page_config(
     page_title="Previsão de Renda",
     page_icon="c:\\Users\\55819\\Downloads\\grafico-histograma.png",
     layout="wide",
)

st.write('# Análise exploratória da previsão de renda')

renda = pd.read_csv('c:\\Users\\55819\\Downloads\\original (1)\\projeto 2\\input\\previsao_de_renda.csv')
st.sidebar.title('Filtros')


renda.data_ref = pd.to_datetime(renda.data_ref)

valor_min = renda.data_ref.min()
valor_max = renda.data_ref.max()

# Filtro de data
data_escolhida = st.sidebar.date_input('Escolha um intervalo de datas', value=[valor_min, valor_max], min_value=valor_min, max_value=valor_max)


# Filtrando os dados pelo intervalo de datas escolhido
dados_filtrados = renda[(renda.data_ref >= pd.to_datetime(data_escolhida[0])) & (renda.data_ref <= pd.to_datetime(data_escolhida[1]))]


import time
with st.spinner('Carregando'):
    if (data_escolhida != valor_min) & (data_escolhida !=valor_max): 
     time.sleep(3)
st.success('Feito !')

st.write('## Gráficos ao longo do tempo')
st.subheader('Posse de imóvel')
st.line_chart(dados_filtrados.groupby(['data_ref', 'posse_de_imovel'])['renda'].mean().unstack())
st.subheader('Posse de veículo')
st.line_chart(dados_filtrados.groupby(['data_ref', 'posse_de_veiculo'])['renda'].mean().unstack())
st.subheader('Quantidade de filhos')
st.bar_chart(dados_filtrados.groupby(['data_ref', 'qtd_filhos'])['renda'].mean().unstack())
st.subheader('Tipo de renda')
st.area_chart(dados_filtrados.groupby(['data_ref', 'tipo_renda'])['renda'].mean().unstack())
st.subheader('Educação')
st.area_chart(dados_filtrados.groupby(['data_ref', 'educacao'])['renda'].mean().unstack())
st.subheader('Estado civil')
st.line_chart(dados_filtrados.groupby(['data_ref', 'estado_civil'])['renda'].mean().unstack())
st.subheader('Tipo de residência')
st.line_chart(dados_filtrados.groupby(['data_ref', 'tipo_residencia'])['renda'].mean().unstack())

st.write('#  Gráficos Bivariados')

left_column, right_column = st.columns(2)
media_renda_por_filho = renda.groupby('qtd_filhos')['renda'].mean()
left_column.bar_chart(media_renda_por_filho,y_label='Renda média',x_label='Quantidade de filhos')
# Vamos calcular a média de renda para cada tipo de renda
media_renda_por_tipo = renda.groupby('tipo_renda')['renda'].mean()
# crie o gráfico de barras
right_column.bar_chart(media_renda_por_tipo,y_label='Renda média',x_label='Tipo de renda')

left_column, right_column = st.columns(2)
media_imovel = renda.groupby('posse_de_imovel')['renda'].mean()
media_veiculo = renda.groupby('posse_de_veiculo')['renda'].mean()
left_column.bar_chart(media_imovel,y_label='Renda média',x_label='Possui imovel')
right_column.bar_chart(media_veiculo,y_label='Renda média',x_label='Possui veiculo')


st.write('# Árvore de Regressão')

# Preprocessamento
label_encoders = {}
for column in ['sexo','qtd_filhos','tipo_renda','idade','renda']:
    le = LabelEncoder()
    renda[column] = le.fit_transform(renda[column])
    label_encoders[column] = le


# Definindo variáveis independentes e dependentes
X = renda[['sexo','qtd_filhos','tipo_renda','idade']]
y = renda['renda']

# Dividindo os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

#treinando as arvores definindo sua profundidade

regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=8)

regr_1.fit(X_train,y_train)
regr_2.fit(X_train,y_train)

# calcular o mse de cada arvore
# para calcular o mse é preciso a previsão então vamos faze-la primeiro

from sklearn.metrics import mean_squared_error

# previsão base teste 
y_train_pred1 = regr_1.predict(X_train)
y_train_pred2 = regr_2.predict(X_train)

# Previsões para a base de testes
y_test_pred1 = regr_1.predict(X_test)
y_test_pred2 = regr_2.predict(X_test)

# Cálculo do MSE para a base de treino
mse_train1 = mean_squared_error(y_train, y_train_pred1)
mse_train2 = mean_squared_error(y_train, y_train_pred2)

# Cálculo do MSE para a base de testes
mse_test1 = mean_squared_error(y_test, y_test_pred1)
mse_test2 = mean_squared_error(y_test, y_test_pred2)

# st.write("MSE Treino - Modelo 1: ", mse_train1)
# st.write("MSE Teste - Modelo 1: ", mse_test1)
# st.write("MSE Treino - Modelo 2: ", mse_train2)
# st.write("MSE Teste - Modelo 2: ", mse_test2)  

path = regr_2.cost_complexity_pruning_path(X_train,y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, impurities)
plt.xlabel("Alpha efetivo")
plt.ylabel("Impureza total das folhas")


clfs = []

for ccp_alpha in ccp_alphas:
    clf = DecisionTreeRegressor(random_state=0,ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

tree_depths = [clf.tree_.max_depth for clf in clfs]
plt.figure(figsize=(10,  6))
plt.plot(ccp_alphas[:-1], tree_depths[:-1])
plt.xlabel("effective alpha")
plt.ylabel("Profundidade da árvore")


train_scores = [mean_squared_error(y_train , clf.predict(X_train)) for clf in clfs]
test_scores  = [mean_squared_error(y_test  , clf.predict(X_test )) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("MSE")
ax.set_title("MSE x alpha do conjunto de dados de treino e teste")
ax.plot(ccp_alphas[:-1], train_scores[:-1], marker='o', label="treino",
        drawstyle="steps-post")
ax.plot(ccp_alphas[:-1], test_scores[:-1], marker='o', label="teste",
        drawstyle="steps-post")
ax.legend()
plt.show()


arvore_final = DecisionTreeRegressor(random_state=0, max_depth=4, min_samples_leaf=10)
arvore_final.fit(X_train,y_train)


st.write(f'profundidade: {arvore_final.tree_.max_depth}')
st.write(f'R-quadrado na base de teste: {arvore_final.score(X_test,y_test)}')
st.write(f'MSE na base de teste: {mean_squared_error(y_test,arvore_final.predict(X_test))}')



# DOT data
from sklearn import tree
dot_data = tree.export_graphviz(arvore_final,
                feature_names=X.columns,
                class_names=['renda'],
                filled=True)

# Draw graph
st.graphviz_chart(dot_data)


