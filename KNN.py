from collections import Counter
"""
Class of KNN classifier.
Implements KKN algorithm.
"""
class KNNClassifier(object):
    def __init__(self,train_examples, train_tags, k):
        """
        Initializes KNNClassifier.
        :param train_examples: list of examples. each example is a list of feature values.
        :param train_tags: list of tags for train_examples
        :param k: Number of k-closest-examples to choose from in prediction.
        """
        self.k = k
        self.train_examples = train_examples
        self.train_tags = train_tags

    def predict(self, example):
        """
        gets an example and returns a predicted tag for the example.
        :param example: list of features values
        :return: prediction
        """
        examples_and_tags = [(ex, tag) for ex, tag in
                             zip(self.train_examples, self.train_tags)]
        distances = []
        # compute hamming distance between the example and all the train examples.
        for example_and_tag in examples_and_tags:
            distance = self.hamming_distance(example_and_tag[0], example)
            distances.append((example_and_tag, distance))

        closest_k = sorted(distances, key=lambda x : x[1])[:self.k]
        # extract just the tags
        closest_k = [item[0][1] for item in closest_k]
        return self.get_common_tag(closest_k)


    def hamming_distance(self, first_example, second_example):
        """
        Computes hamming distance (difference between feature values) between
        first_example and second_example
        :param first_example: first example
        :param second_example: second example
        :return: hamming distance
        """
        distance = 0
        for feature_1, feature_2 in zip(first_example, second_example):
            if feature_1 != feature_2:
                distance += 1

        return distance


    def get_common_tag(self, tags):
        """
        Returns the most common tag in tags.
        :param tags: tags list
        :return: most common tag.
        """
        tags_counter = Counter()
        for tag in tags:
            tags_counter[tag] += 1

        return tags_counter.most_common(1)[0][0]
