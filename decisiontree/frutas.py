# %%
import pandas as pd
from sklearn import tree

df = pd.read_excel("../data/dados_frutas.xlsx")
df
# %%

filtro_redonda = df["Arredondada"] == 1
filtro_suculenta = df["Suculenta"] == 1
filtro_vermelha = df["Vermelha"] == 1
filtro_doce = df["Doce"] == 1


df[filtro_redonda & filtro_suculenta & filtro_vermelha & filtro_doce]

# %%


features = ["Arredondada",	"Suculenta", "Vermelha", "Doce"]
target = "Fruta"

x = df[features]
y = df[target]

# %%

arvore = tree.DecisionTreeClassifier()
arvore.fit(x, y)

# %%
import matplotlib.pyplot as plt

plt.figure(dpi= 600)
tree.plot_tree(arvore, 
               class_names=arvore.classes_, 
               feature_names=features,
               filled=True)

# %%

arvore.predict([[0,1,1,1]])

# %%
probas = arvore.predict_proba([[1,1,1,1]])[0]
pd.Series(probas, index=arvore.classes_)

# %%
