from collections import Counter
import math
import Utils as ut

class DecisionTreeClassifier(object):
    """
    Class of Decision Tree Classifier.
    Implements DTL algorithm.
    """
    def __init__(self, features, train_examples, train_tags):
        """
        Initializes DecisionTreeClassifier.
        :param features: features list (including target feature)
        :param train_examples: train examples
        :param train_tags: train tags
        """
        self.features = features
        self.train_examples = train_examples
        self.train_tags = train_tags
        self.feature_domain_dict = self.get_feature_domain_dict()
        tagged_examples = [(example, tag) for example, tag in zip(self.train_examples, self.train_tags)]
        self.decisionTree = DecisionTree(self.DTL(tagged_examples, features, 0,self.get_default_tag(train_tags)), features)


    def DTL(self, examples_and_tags, features, depth, default=None):
        """
        DTL algorithm for constructing decision tree.
        :param examples_and_tags: list of (example, tag) tuples
        :param features: unselected features list
        :param depth: depth of the current node. should firstly be 0.
        :param default: default prediction
        :return: root of the decision tree
        """
        # examples is an empty list
        if not examples_and_tags:
            return DecisionTreeNode(None, depth, is_leaf=True, pred=default)

        # all examples have the same tag
        tags = [tag for example, tag in examples_and_tags]
        examples = [example for example, tag in examples_and_tags]

        if len(set(tags)) == 1:
            return DecisionTreeNode(None, depth, is_leaf=True, pred=tags[0])

        # features list is empty
        if not features:
            return DecisionTreeNode(None, depth, is_leaf=True,  pred=self.get_default_tag(tags))

        # find the best feature for the current node
        best_feature = self.choose_feature(features, examples, tags)
        feature_index = self.get_feature_index(best_feature)
        node = DecisionTreeNode(best_feature, depth)
        child_features = features[:]
        child_features.remove(best_feature)
        for possible_value in self.feature_domain_dict[best_feature]:
            examples_and_tags_vi = [(example, tag) for example,tag in zip(examples, tags)
                                  if example[feature_index] == possible_value]
            # create child subtree for every possibe value of the feature
            child = self.DTL(examples_and_tags_vi, child_features, depth + 1, self.get_default_tag(tags))
            node.children[possible_value] = child

        return node


    def choose_feature(self, features, examples, tags):
        """
        Chooses the best feature according to the given examples and tags.
        The feature that have the highest gain will be chosen.
        :param features: list of features
        :param examples: examples
        :param tags: tags
        :return: best feature
        """
        features_gains_dict = {feature : self.get_gain(examples, tags, feature) for feature in features}
        max_gain = 0
        max_feature = features[0]
        for feature in features:
            if features_gains_dict[feature] > max_gain:
                max_gain = features_gains_dict[feature]
                max_feature = feature

        # return the feature with the highest gain
        return max_feature

    def calculate_entropy(self, tags):
        """
        Calculates the entropy of the given tags
        :param tags:
        :return:
        """
        tags_counter = Counter()

        if not tags:
            return 0

        for tag in tags:
            tags_counter[tag] += 1
        classes_probs = [tags_counter[tag] / float(len(tags)) for tag in tags_counter]
        if 0.0 in classes_probs:
            return 0

        entropy = 0
        for prob in classes_probs:
            entropy -= prob * math.log(prob, 2)

        return entropy

    def get_gain(self, examples, tags, feature):
        """
        Computes the gain for the given feature according to the given examples and tags.
        :param examples: examples
        :param tags: tags
        :param feature: feature
        :return: gain of the feature.
        """
        initial_entropy = self.calculate_entropy(tags)
        relative_entropy_per_feature = []
        feature_index = self.get_feature_index(feature)
        for possible_value in self.feature_domain_dict[feature]:
            examples_and_tags_vi = [(example, tag) for example, tag in zip(examples, tags)
                                            if example[feature_index] == possible_value]
            tags_vi = [tag for example, tag in examples_and_tags_vi]
            entropy_vi = self.calculate_entropy(tags_vi)
            if not examples:
                pass
            relative_entropy = (float(len(examples_and_tags_vi)) / len(examples)) * entropy_vi
            relative_entropy_per_feature.append(relative_entropy)

        return initial_entropy - sum(relative_entropy_per_feature)



    def get_feature_index(self, feature):
        """
        return the feature index in self.features.
        :param feature: feature
        :return: feature's index
        """
        return self.features.index(feature)

    def get_feature_domain_dict(self):
        """
        Creates and returns a dict in which the key is a feature and the value is a list of all the
        possible values of the feature.
        :return: dict
        """
        feature_domain_dict = {}
        for feature_index in range(len(self.train_examples[0])):
            domain = set([example[feature_index] for example in self.train_examples])
            feature_domain_dict[self.features[feature_index]] = domain

        return feature_domain_dict


    def get_default_tag(self, tags):
        """
        returns the most common tag in tags list.
        :param tags: tags list
        :return:
        """
        tags_counter = Counter()
        for tag in tags:
            tags_counter[tag] += 1

        if len(tags_counter) == 2 and list(tags_counter.values())[0] == list(tags_counter.values())[1]:
            return ut.find_positive_tag(tags_counter.keys())

        return tags_counter.most_common(1)[0][0]

    def predict(self, example):
        """
        gets an example and returns a predicted tag for the example.
        :param example: list of features values
        :return: prediction
        """
        return self.decisionTree.traverse_tree(example)

    def write_tree_to_file(self, output_file_name):
        """
        writes the tree structure to output_file_name
        :param output_file_name: output file name
        """
        with open(output_file_name, "w") as output:
            tree_string = self.decisionTree.get_tree_string(self.decisionTree.root)
            output.write(tree_string[:len(tree_string) - 1])


class DecisionTree(object):
    """
    Class of decision tree.
    Holds the tree's root and traverse tree functionality.
    """
    def __init__(self, root, features):
        """
        Initializes DecisionTree class.
        :param root: DecisionTreeNode root of the tree
        :param features: features list
        """
        self.root = root
        self.features = features

    def traverse_tree(self, example):
        """
        return the prediction for the example after traversing the tree and reaching to a leaf
        according to the example features' values.
        :param example: example
        :return: prediction
        """
        current_node = self.root
        while not current_node.is_leaf:
            feature_value = example[self.get_feature_index(current_node.feature)]
            current_node = current_node.children[feature_value]

        return current_node.pred

    def get_feature_index(self, feature):
        """
        return the feature index in self.features.
        :param feature: feature
        :return: feature's index
        """
        return self.features.index(feature)

    def get_tree_string(self, node):
        """
        return the string of the tree structure.
        :param node: start node
        :return: tree as string
        """
        string = ""
        for child in sorted(node.children):
            string += node.depth * "\t"
            if node.depth > 0:
                string += "|"
            string += node.feature + "=" + child
            if node.children[child].is_leaf:
                string += ":" + node.children[child].pred + "\n"
            else:
                string += "\n" + self.get_tree_string(node.children[child])

        return string


class DecisionTreeNode(object):
    """
    Class for Decision Tree Node.
    Each non-leaf node contains the feature, depth and children dict that maps every possible value
    of the feature to a child node.
    Each leaf node contains pred attribute.
    """
    def __init__(self, feature, depth, is_leaf=False, pred=None):
        """
        Initializes DecisionTreeNode.
        :param feature: feature
        :param depth: depth of the node in the decision tree.
        :param is_leaf: is the node leaf or not
        :param pred: possible tag
        """
        self.feature = feature
        self.depth = depth
        self.is_leaf = is_leaf
        self.pred = pred
        self.children = {}