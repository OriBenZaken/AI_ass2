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

