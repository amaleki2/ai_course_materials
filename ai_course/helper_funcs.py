import graphviz
from sklearn.tree import export_graphviz
from subprocess import check_call
from PIL import Image


def plot_decision_tree(decision_tree, feature_names, class_names):
  with open("tree.dot", 'w') as f:
     f = export_graphviz(decision_tree,
                        max_depth = 3,
                        out_file = f,
                        impurity = True,
                        feature_names = feature_names,
                        class_names = class_names,
                        rounded = True,
                        filled = True )
  check_call(['dot','-Tpng','tree.dot','-o','tree.png'])
  img = Image.open("tree.png")
  img.save("tree.png")