# Homework 2
# name: Martine De Cock
# description: Training and testing decision trees with discrete-values attributes

import sys
import math
import pandas as pd

class DecisionNode:

    # A DecisionNode contains an attribute and a dictionary of children.
    # The attribute is either the attribute being split on, or the predicted label if the node has no children.
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = {}

    # Visualizes the tree
    def display(self, level = 0):
        if self.children == {}: # reached leaf level
            print(": ", self.attribute, end="")
        else:
            for value in self.children.keys():
                prefix = "\n" + " " * level * 4
                print(prefix, self.attribute, "=", value, end="")
                self.children[value].display(level + 1)

    # Predicts the target label for instance x
    def predicts(self, x):
        if self.children == {}: # reached leaf level
            return self.attribute
        value = x[self.attribute]
        subtree = self.children[value]
        return subtree.predicts(x)


# Illustration of functionality of DecisionNode class
def funTree():
    myLeftTree = DecisionNode('humidity')
    myLeftTree.children['normal'] = DecisionNode('no')
    myLeftTree.children['high'] = DecisionNode('yes')
    myTree = DecisionNode('wind')
    myTree.children['weak'] = myLeftTree
    myTree.children['strong'] = DecisionNode('no')
    return myTree

def entropy(lists):
    counts = lists.value_counts()
    counts = counts.to_frame(name='count').reset_index()
    counts = counts.pivot_table(index=str(target), columns=counts.columns[0], values='count').fillna(0)
    n = counts.iloc[0,0]
    y = counts.iloc[1,0]
    s = n + y
    nlog = 0
    ylog = 0
    if y == 0:
        ylog = 0
    else:
        ylog = y / s * math.log2(y / s)
    if n == 0:
        nlog = 0
    else:
        nlog = n / s * math.log2(n / s)
    ent = - (ylog + nlog)
    return ent

def gain_calc(lists, target, ent_prior, s):
    counts = lists.value_counts()
    counts = counts.to_frame(name='count').reset_index()
    counts = counts.pivot_table(index=str(target), columns=counts.columns[0], values='count').fillna(0)
    ent = 0
    for column in counts:
        n = counts[column].values[0]
        y = counts[column].values[1]
        # s = n + y
        nlog = 0
        ylog = 0
        if y == 0:
            ylog = 0
        else:
            ylog = y / s * math.log2(y / s)
        if n == 0:
            nlog = 0
        else:
            nlog = n / s * math.log2(n / s)
        ent = ent - (ylog + nlog)
    gain = ent_prior - ent
    return gain
def id3(examples, target, attributes):
    target_data = examples[str(target)].value_counts().to_dict()
    sample_ent = entropy(examples[str(target)])
    maxent = 0
    target_val = []
    # for ind in columns in
    list_branch = examples.groupby([str(attributes[3])])[str(target)]
    print(attributes[3])
    temp = gain_calc(list_branch, target, sample_ent, examples.shape[0])
    tree = temp
    return temp


####################   MAIN PROGRAM ######################

# Reading input data
train = pd.read_csv('playtennis_train.csv')
test = pd.read_csv('playtennis_test.csv')
target = 'playtennis'
attributes = train.columns.tolist()
attributes.remove(target)

# Learning and visualizing the tree
tree = id3(train,target,attributes)
# tree.display()
print(tree)

# Evaluating the tree on the test data
# correct = 0
# for i in range(0,len(test)):
#     if str(tree.predicts(test.loc[i])) == str(test.loc[i,target]):
#         correct += 1
# print("\nThe accuracy is: ", correct/len(test))