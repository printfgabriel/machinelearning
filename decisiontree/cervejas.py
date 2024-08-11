# %%
import pandas as pd
from sklearn import tree
pd.set_option("future.no_silent_downcasting", True)


df = pd.read_excel("../data/dados_cerveja.xlsx")
df
# %%
features = ["temperatura", "copo", "espuma", "cor"]
target = "classe"

x = df[features]
y = df[target]

# %%

x = x.replace({
    "mud":1, "pint":0,
    "sim":1, "não":0,
    "escura":1, "clara":0
}).astype(int)
x
# %%
arvore = tree.DecisionTreeClassifier()
arvore.fit(x, y)
# %%
import matplotlib.pyplot as plt

plt.figure(dpi= 600)

tree.plot_tree(arvore,
              class_names = arvore.classes_,
              feature_names = features,
              filled = True)

# %%
# "temperatura", "copo", "espuma", "cor"
# probas = arvore.predict_proba([[-1, 1, 0, 1]])[0]
input_data = pd.DataFrame([[-1, 1, 0, 1]], columns=features)
probas = arvore.predict_proba(input_data)[0]
pd.Series(probas, index=arvore.classes_)
#plt.show()
# %%
