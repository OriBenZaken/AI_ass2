import Utils as ut
from functools import reduce
"""
Class of Naive Bayes Classifier.
Implements Naive Bayes ML algorithm with smoothing.
"""
class NaiveBayesClassifier(object):
    def __init__(self, train_examples, train_tags):
        """
        Initializes NayesBayesClassifier
        :param train_examples: train examples
        :param train_tags: train tags
        """
        self.train_examples = train_examples
        self.train_tags = train_tags
        self.examples_by_tag_dict = self.get_examples_by_tag_dict()
        self.feature_domain_size_dict = self.get_feature_domain_size_dict()


    def get_examples_by_tag_dict(self):
        """
        Creates and returns a dict in which the key is a tag and
        the value is list of the examples that have this tag.
        :return: dict
        """
        examples_by_tag_dict = {}
        for example, tag in zip(self.train_examples, self.train_tags):
            if tag in examples_by_tag_dict:
                examples_by_tag_dict[tag].append(example)
            else:
                examples_by_tag_dict[tag] = [example]

        return examples_by_tag_dict


    def get_feature_domain_size_dict(self):
        """
        Creates and returns a dict in which the key is a feature index and
        the value is the suze of the feature domain (all possible values the feature can get).
        :return: dict
        """
        feature_domain_size_dict = {}
        for feature_index in range(len(self.train_examples[0])):
            domain = set([example[feature_index] for example in self.train_examples])
            feature_domain_size_dict[feature_index] = len(domain)

        return feature_domain_size_dict


    def predict(self, example):
        """
        gets an example and returns a predicted tag for the example.
        :param example: list of features values
        :return: prediction
        """
        max_prob = 0
        max_tag = list(self.examples_by_tag_dict.keys())[0]
        probs = []
        # calculates the probability for each class, keep tracking the maximum prob
        for tag in self.examples_by_tag_dict:
            prob = self.calculate_prob(example, self.examples_by_tag_dict[tag])
            probs.append(prob)
            if prob > max_prob:
                max_prob, max_tag = prob, tag

        if len(probs) == 2 and probs[0] == probs[1]:
            return ut.find_positive_tag(self.examples_by_tag_dict.keys())

        # return the tag with the highest probability
        return max_tag


    def calculate_prob(self, example, tag_group):
        """
        calculates the probability that the example belongs to the class of the examples in tag_group
        :param example: example
        :param tag_group: list of examples that have the same tag.
        :return: probability for class
        """
        conditioned_prob_list = []
        tag_group_size = len(tag_group)
        # computes conditioned probability for each feature
        for feature_index in range(len(example)):
            f_counter = 1
            domain_size = self.feature_domain_size_dict[feature_index]
            for train_example in tag_group:
                if train_example[feature_index] == example[feature_index]:
                    f_counter += 1
            conditioned_prob_list.append(float(f_counter)/ (tag_group_size + domain_size))
        class_prob = float(len(tag_group)) / len(self.train_examples)

        return reduce(lambda x, y: x * y, conditioned_prob_list) * class_prob