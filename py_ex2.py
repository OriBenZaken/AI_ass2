import  Utils as ut
from KNN import KNNClassifier
from NaiveBayes import NaiveBayesClassifier
from DecisionTree import DecisionTreeClassifier

def write_output_files(test_tags, preds_per_classifier, accuracy_per_classifier, dt_cls):
    """
    Writes the classifiers results on the test set to output.txt and output_tree.txt file.
    :param test_tags: tast tags
    :param preds_per_classifier: list of predictions of each classifier
    :param accuracy_per_classifier: list of accuracy of each classifier
    :param dt_cls: DecisionTreeClassifier object.
    """
    dt_preds, knn_preds, nb_preds = preds_per_classifier[0], preds_per_classifier[1], preds_per_classifier[2]
    dt_acc, knn_acc, nb_acc = accuracy_per_classifier[0], accuracy_per_classifier[1], accuracy_per_classifier[2]
    with open("output.txt", "w") as output:
        lines = []
        lines.append("Num\tDT\tKNN\tnaiveBayes")
        i = 1
        for true_tag, dt_pred, knn_pred, nb_pred in zip(test_tags, dt_preds, knn_preds, nb_preds):
            lines.append("{}\t{}\t{}\t{}".format(i, dt_pred, knn_pred, nb_pred))
            i += 1
        lines.append("\t{}\t{}\t{}".format(dt_acc, knn_acc, nb_acc))
        output.writelines("\n".join(lines))

    dt_cls.write_tree_to_file("output_tree.txt")


def main():
    """
    Main function of ex 2.
    Runs KNN, Naive Bayes and Decision Tree classifiers and writes the
    results to output.txt and output_tree.txt
    """
    features, train_examples, train_tags = ut.read_labled_file("TESTS-EX2-V3/TEST1/train.txt")
    _,test_examples, test_tags = ut.read_labled_file("TESTS-EX2-V3/TEST1/test.txt")

    knn_cls = KNNClassifier(train_examples, train_tags, k=5)
    nb_cls = NaiveBayesClassifier(train_examples, train_tags)
    dt_cls = DecisionTreeClassifier(features[:len(features) - 1], train_examples, train_tags)

    classifiers = [dt_cls, knn_cls, nb_cls]
    preds_per_classifier = []
    accuracy_per_classifier = []

    for classifier in classifiers:
        preds = []
        for example, tag in zip(test_examples, test_tags):
            pred = classifier.predict(example)
            preds.append(pred)
        preds_per_classifier.append(preds)
        accuracy_per_classifier.append(ut.get_accuracy(test_tags, preds))

    write_output_files(test_tags, preds_per_classifier, accuracy_per_classifier, dt_cls)

if __name__ == "__main__":
    main()