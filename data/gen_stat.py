import os

with open("my_train_food.txt", "r") as myr:
    train_info = myr.readlines()
with open("my_test_food.txt", "r") as mye:
    test_info = mye.readlines()

train_labels, test_labels = [], []

for line in train_info:
    _, label = line.strip().split(".jpg ")
    if label + "\n" not in train_labels:
        train_labels.append(label + "\n")

for line in test_info:
    _, label = line.strip().split(".jpg ")
    if label + "\n" not in test_labels:
        test_labels.append(label + "\n")

assert len(train_labels) == len(test_labels)

for label in train_labels:
    assert label in test_labels
for label in test_labels:
    assert label in train_labels

with open("stat_food.txt", "w") as sat:
    sat.writelines(train_labels)
