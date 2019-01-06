from collections import Counter

class KNNClassifier(object):
    def __init__(self,train_examples, train_tags, k):
        self.k = k
        self.train_examples = train_examples
        self.train_tags = train_tags

    def predict(self, example):
        examples_and_tags = [(ex, tag) for ex, tag in
                             zip(self.train_examples, self.train_tags)]
        distances = []
        for example_and_tag in examples_and_tags:
            distance = self.hamming_distance(example_and_tag[0], example)
            distances.append((example_and_tag, distance))

        closest_k = sorted(distances, key=self.extract_distance)[:self.k]
        # extract just the tags
        closest_k = [item[0][1] for item in closest_k]
        return self.get_common_tag(closest_k)


    def hamming_distance(self, first_example, second_example):
        distance = 0
        for feature_1, feature_2 in zip(first_example, second_example):
            if feature_1 != feature_2:
                distance += 1

        return distance

    def extract_distance(self, tagged_example_and_distace):
        return  tagged_example_and_distace[1]

    def get_common_tag(self, tags):
        tags_counter = Counter()
        for tag in tags:
            tags_counter[tag] += 1

        return tags_counter.most_common(1)[0][0]
