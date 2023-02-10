from random import shuffle

import numpy as np


def split_data(csv_path, train=0.7, total=1):
    csv_path = csv_path[:-4]
    with open(csv_path + ".csv", "r") as file:
        data = file.readlines()
    data = np.asarray(data[1:])
    print(data)
    data = data[:int(len(data) * total)]

    indices = list(range(len(data)))
    shuffle(indices)
    k = int(train * len(indices))
    train_set = data[indices[:k]]
    test_set = data[indices[k:]]

    with open(csv_path + "_train.csv", "w") as file:
        for line in train_set:
            file.write(line)
    with open(csv_path + "_test.csv", "w") as file:
        for line in test_set:
            file.write(line)


if __name__ == "__main__":
    split_data(".\\datasets\\songs\\piki_dataset.csv")
