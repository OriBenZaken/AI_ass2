from collections import Counter
import math

class DecisionTreeClassifier(object):
    def __init__(self, features, train_examples, train_tags):
        self.features = features
        self.train_examples = train_examples
        self.train_tags = train_tags
        self.feature_domain_dict = self.get_feature_domain_dict()

    def DTL(self, examples_and_tags, features, depth, default=None):
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
            return DecisionTreeNode(None, depth, is_leaf=True, pred=self.get_default_tag(tags))

        best_feature = self.choose_feature(features, examples, tags)
        pass

    def choose_feature(self, features, example, tags):
        pass

    def calculate_entropy(self, tags):
        tags_counter = Counter()
        for example, tag in tags:
            tags_counter[tag] += 1
        classes_probs = [tags_counter[tag] / float(len(tags)) for tag in tags_counter]
        entropy = 0
        for prob in classes_probs:
            entropy -= -prob * math.log(prob, 2)

    def get_gain(self, examples, tags, feature):
        initial_entropy = self.calculate_entropy(tags)
        relative_entropy_per_feature = []
        feature_index = self.get_feature_index(feature)
        for possible_value in self.feature_domain_dict[feature]:
            examples_and_tags_vi = [(example, tag) for example, tag in zip(examples, tags)
                                            if example[feature_index] == possible_value]
            tags_vi = [tag for example, tag in examples_and_tags_vi]
            entropy_vi = self.calculate_entropy(tags_vi)
            relative_entropy = (float(len(examples_and_tags_vi)) / len(examples)) * entropy_vi
            relative_entropy_per_feature.append(relative_entropy)

        return initial_entropy -sum(relative_entropy_per_feature)



    def get_feature_index(self, feature):
        return self.features.index(feature)

    def get_feature_domain_dict(self):
        feature_domain_dict = {}
        for feature_index in range(len(self.train_examples[0])):
            domain = set([example[feature_index] for example in self.train_examples])
            feature_domain_dict[self.features[feature_index]] = domain

        return feature_domain_dict


    def get_default_tag(self, tags):
        tags_counter = Counter()
        for tag in tags:
            tags_counter[tag] += 1

        # todo: return positive if yes = no
        return tags_counter.most_common(1)[0][0]


class DecisionTree(object):
    def __init__(self, root):
        self.root = root


class DecisionTreeNode(object):
    def __init__(self, feature, depth, is_leaf=False, pred=None):
        self.feature = feature
        self.depth = depth
        self.is_leaf = is_leaf
        self.pred = pred
        self.children = {}