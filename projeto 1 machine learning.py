#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



# In[26]:


df=  pd.read_csv('casakingcout.csv')


# In[4]:


df.describe()


# In[27]:


df.dtypes


# In[28]:


#removendo variáveis
dadoslimpo=df.drop(['id','date','waterfront'],axis=1)


# In[29]:


#defenindo atributos e alvo
x=dadoslimpo.drop('price',axis=1)
y=dadoslimpo['price']


# In[30]:


#avaliando o valor P e R^2 para saber quais variaveis sao insignificantes para o meu modelo
import statsmodels.api as sm
xc=sm.add_constant(x)
modelo_v1=sm.OLS(y,xc)
modelo_v2=modelo_v1.fit()
modelo_v2.summary()


# In[31]:


dadoslimpo.drop(['floors'] , axis=1)


# In[32]:


# ANALISANDO MULTICOLINEARIDADE

correlation_matrix = df.corr(numeric_only=True)
# Verificando as correlações fortes 
strong_correlations = (correlation_matrix > 0.8) & (correlation_matrix < 1.0)
# Identificando as criáveis com multicolinearidade
multicollinear_vars = set()
for col in strong_correlations.columns:
    correlated_vars = strong_correlations.index[strong_correlations[col]]
    if len(correlated_vars) > 1:
        multicollinear_vars.update(correlated_vars)

if multicollinear_vars:
    print("\nVariáveis com multicolinearidade:{}".format(multicollinear_vars))
    
else:
    print("\nNão foi encontrada multicolinearidade entre as variáveis.")


# In[42]:


#definindo variáveis atributo e variável alvo
x=dadoslimpo.drop('price',axis=1)
y=dadoslimpo['price']


# In[43]:


# Dividindo os dados em conjuntos de treinamento e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.30, random_state=10)
# Treinar o modelo usando os dados de treinamento
modelo=LinearRegression()
modelo.fit(x_treino, y_treino)


# In[44]:


resultado= modelo.score(x_teste, y_teste)
print(resultado)

