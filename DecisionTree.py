class DecisionTreeClassifier(object):
    pass

class DecisionTree(object):
    def __init__(self, root):
        self.root = root


class DecisionTreeNode(object):
    def __init__(self, feature, value, is_leaf=False, pred=None):
        self.feature = feature
        self.value = value
        self.is_leaf = is_leaf
        self.pred = pred
        self.children = []

    def add_feature_child(self, feature, value):
        node = DecisionTreeNode(feature, value)
        self.children.append(node)
        return node

    def add_leaf(self, pred):
        leaf = DecisionTreeNode(None, None, is_leaf=True, pred=pred)
        self.children.append(leaf)
