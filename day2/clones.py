# %%
import pandas as pd
df = pd.read_parquet("../data/dados_clones.parquet")
df

# %%

# Searching for the problem: comparing data between aptos and defeituosos
df.groupby(["Status "])[['Estatura(cm)', 'Massa(em kilos)']].mean()

# %%
df['Status_bool'] = df['Status '] == 'Apto'
df
# %%

df.groupby(["Distância Ombro a ombro"])[['Status_bool']].mean()
df.groupby(["Tamanho do crânio"])[['Status_bool']].mean()
df.groupby(["Tamanho dos pés"])[['Status_bool']].mean()

# %%
# Found some problem here: Yoda's and Shaak Ti's squad area dying a lot!!!!
df.groupby(["General Jedi encarregado"])[['Status_bool']].mean()

# %%

features = ["Estatura(cm)", 
            "Massa(em kilos)", 
            "Distância Ombro a ombro", 
            "Tamanho do crânio",
            "Tamanho dos pés"]

x = df[features]
x # still need to put features as binary, here we go:

from feature_engine import encoding

cat_features = ["Distância Ombro a ombro", 
                "Tamanho do crânio",
                "Tamanho dos pés"]

# Transformar variaveis categoricas em numericas
onehot = encoding.OneHotEncoder(variables=cat_features)
onehot.fit(x)
x = onehot.transform(x)
x

# %%
from sklearn import tree

# here we can see that clones with some specific weight and height are  more likely to be defective.
# This conclusion is different from that analysis we made on hand before. maybe those generals received clones with bad features
arvore = tree.DecisionTreeClassifier(max_depth=3)
arvore.fit(x, df["Status "])

# %%
import matplotlib.pyplot as plt
plt.figure(dpi=600)

tree.plot_tree(arvore, 
               class_names=arvore.classes_,
               feature_names=x.columns,
               filled = True,
               )

plt.show()
# %%