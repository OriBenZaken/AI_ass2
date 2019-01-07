def read_labled_file(labled_file_name):
    features = []
    examples = []
    tags = []
    with open(labled_file_name, "r") as file:
        content = file.readlines()
        features += content[0].strip("\n").strip().split("\t")

        for line in content[1:]:
            line = line.strip("\n").strip().split("\t")
            example , tag = line[:len(line) - 1], line[-1]
            examples.append(example)
            tags.append(tag)

    return features, examples, tags

def get_accuracy(true_tags, predicted_tags):
    good = bad = 0.0
    for true_tag, pred in zip(true_tags, predicted_tags):
        if true_tag == pred:
            good += 1
        else:
            bad += 1
    return round((good/(good + bad)) * 100, 2)

def find_positive_tag(tags):
    for tag in tags:
        if tag in ["yes", "true"]:
            return tag

    return tag[0]