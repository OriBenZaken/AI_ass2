import  Utils as ut
from KNN import KNNClassifier
from NaiveBayes import NaiveBayesClassifier

def main():
    features, train_examples, train_tags = ut.read_labled_file("train.txt")
    _,test_examples, test_tags = ut.read_labled_file("test.txt")

    # classifier = KNNClassifier(train_examples, train_tags, k=5)
    classifier = NaiveBayesClassifier(train_examples, train_tags)

    for example, tag in zip(test_examples, test_tags):
        pred = classifier.predict(example)
        print("example: {} , true tag: {}, predicted tag: {}".format(example, tag, pred))

if __name__ == "__main__":
    main()