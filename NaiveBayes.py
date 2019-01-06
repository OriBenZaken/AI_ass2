class NaiveBayesClassifier(object):
    def __init__(self, train_examples, train_tags):
        self.train_examples = train_examples
        self.train_tags = train_tags
        self.examples_by_tag_dict = self.get_examples_by_tag_dict()
        self.feature_domain_size_dict = self.get_feature_domain_size_dict()
        pass

    def get_examples_by_tag_dict(self):
        examples_by_tag_dict = {}
        for example, tag in zip(self.train_examples, self.train_tags):
            if tag in examples_by_tag_dict:
                examples_by_tag_dict[tag].append(example)
            else:
                examples_by_tag_dict[tag] = [example]

        return examples_by_tag_dict

    def get_feature_domain_size_dict(self):
        feature_domain_size_dict = {}
        for feature_index in range(len(self.train_examples[0])):
            domain = set([example[feature_index] for example in self.train_examples])
            feature_domain_size_dict[feature_index] = len(domain)

        return feature_domain_size_dict


    def predict(self, example):
        max_prob = 0
        max_tag = self.examples_by_tag_dict.keys()[0]

        for tag in self.examples_by_tag_dict:
            prob = self.calculate_prob(example, self.examples_by_tag_dict[tag])
            if prob > max_prob:
                max_prob, max_tag = prob, tag

        return max_tag


    def calculate_prob(self, example, tag_group):
        conditioned_prob_list = []
        tag_group_size = len(tag_group)
        for feature_index in range(len(example)):
            f_counter = 1
            domain_size = self.feature_domain_size_dict[feature_index]
            for train_example in tag_group:
                if train_example[feature_index] == example[feature_index]:
                    f_counter += 1
            conditioned_prob_list.append(float(f_counter)/ (tag_group_size + domain_size))

        return reduce(lambda x, y: x * y, conditioned_prob_list)