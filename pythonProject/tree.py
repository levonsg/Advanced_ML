import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

dict_tree = {
'feature' : None,
'value' : None,
'true' : None,
'false' : None,
}


X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24)
#df = pd.DataFrame(X_train, columns=['A', 'B', 'C', 'D'])

print(X_train[:,0].min())
thresh = np.linspace(X_train[0].min(), X_train[0].max(), 20)


dict_tree_list = []
for i in thresh:
    dict_tree['value'] = i
    dict_tree['feature'] = 'A'
    (cl, num) = np.unique(y_train[X_train[:,0]>=i], return_counts=True)
    dd = {}
    for j, iter in enumerate(cl):
        dd[j] = num[iter]
    dd = dd.copy()
    dict_tree['true'] = dd

    (cl, num) = np.unique(y_train[X_train[:, 0] < i], return_counts=True)
    ddd = {}
    for j, iter in enumerate(cl):
        ddd[j] = num[iter]
    ddd = ddd.copy()
    dict_tree['false'] = ddd

    dict_tree = dict_tree.copy()
    dict_tree_list.append(dict_tree)

print(dict_tree_list)




