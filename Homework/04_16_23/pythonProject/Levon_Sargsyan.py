import argparse
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

max_depth = 4
criterion = 'gini'
leaf_nodes = None
random_state = 1234

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--target', type=str, required=True)
parser.add_argument('--max_depth', type=int, default=max_depth)
parser.add_argument('--criterion', type=str, default=criterion)
parser.add_argument('--max_leaf_nodes', type=int, default=leaf_nodes)
parser.add_argument('--random_state', type=int, default=random_state)

args = parser.parse_args()

data = pd.read_csv(args.data_path)

X = data.drop(columns=[args.target])
X = X.select_dtypes(exclude=['object'])

y = data[args.target]

clf = DecisionTreeClassifier(max_depth=args.max_depth, criterion=args.criterion, max_leaf_nodes=args.max_leaf_nodes,
                             random_state=args.random_state)
clf.fit(X, y)


plt.figure()
plot_tree(clf, filled=True)
plt.savefig('tree.png', format="png", dpi=700)
