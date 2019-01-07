import  Utils as ut
from KNN import KNNClassifier
from NaiveBayes import NaiveBayesClassifier
from DecisionTree import DecisionTreeClassifier

def main():
    features, train_examples, train_tags = ut.read_labled_file("train.txt")
    _,test_examples, test_tags = ut.read_labled_file("test.txt")

    #classifier = KNNClassifier(train_examples, train_tags, k=5)
    #classifier = NaiveBayesClassifier(train_examples, train_tags)
    classifier = DecisionTreeClassifier(features[:len(features) - 1], train_examples, train_tags)

    preds = []
    for example, tag in zip(test_examples, test_tags):
        pred = classifier.predict(example)
        print("example: {} , true tag: {}, predicted tag: {}".format(example, tag, pred))
        preds.append(pred)

    print(ut.get_accuracy(test_tags, preds))
    classifier.write_tree_to_file("my_output_tree.txt")

if __name__ == "__main__":
    main()