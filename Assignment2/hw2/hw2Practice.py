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

def entropy(y, n):
    s = n + y
    if y == 0:
        ylog = 0
    else:
        ylog = y / s * math.log2(y / s)
    if n == 0:
        nlog = 0
    else:
         nlog = n / s * math.log2(n / s)
    ent = -(ylog + nlog)
    return ent



def gain_calc(group, target, ent_prior, s_tot, flag):
    if flag == 0:
        counts = group.value_counts()
        counts = counts.to_frame(name='count').reset_index()
        counts = counts.pivot_table(index=str(target), columns=counts.columns[0], values='count').fillna(0)
        ent = 0
        ent_branch = 0
        for column in counts:
            n = counts[column].values[0]
            y = counts[column].values[1]
            s = y + n
            ent_branch = s / s_tot * entropy(y, n)
            ent = ent + ent_branch

        gain = ent_prior - ent
    else:
        # DEVICE WAY FOR GAIN CALCULATION FOR BRANCH ITEMS
        gain = 1
    return gain

def MyTree(examples,attributes,  branch, target):
    my_tree = DecisionNode(str(branch))
    branch_bool = examples.groupby([str(branch)])[str(target)].value_counts().to_frame(name='count').reset_index()
    branch_bool = branch_bool.pivot_table(index=str(target), columns=branch_bool.columns[0], values='count').fillna(0)
    df_ent = pd.DataFrame()
    # calculate branch entropy
    i = 0
    new_branch = []
    for column in branch_bool:
        n = branch_bool[column].values[0]
        y = branch_bool[column].values[1]
        ent = entropy(y,n)
        if ent == 0:
            leaf = ''
            if y > n:
                leaf = 'yes'
            else:
                leaf = 'no'
            my_tree.children[str(column)] = DecisionNode(leaf)
            branch_bool = branch_bool.drop(str(column), axis=1)
        else:
            df_ent[str(column)] = [ent]
            i += 1
    group1 = pd.DataFrame()
    for e in df_ent:
        gain = 0
        for name in attributes:
            branch_group = examples.groupby([str(branch), str(name)])[str(target)].value_counts().to_frame(name='count').reset_index()

            group1 = branch_group[branch_group[str(branch)] == e]
            group1 = group1.groupby([str(target)])['count'].sum()
            y = group1.values[0]
            s_tot = group1.sum(axis = 0)
            ent_prior = df_ent.iloc[0][str(e)]
            temp = gain_calc(group1, target, ent_prior, s_tot, 1)
            if temp > gain:
                gain = temp
    # DEVICE WAY FOR GAIN CALCULATION FOR BRANCH ITEMS
    x
    return gain

# Begin IDS algorithm
def id3(examples, target, attributes):
    col_names = attributes
    target_data = examples[str(target)].value_counts()
    target_data = target_data.to_frame(name='count').reset_index()
    target_data = target_data.pivot_table(index=str(target), columns=target_data.columns[0], values='count').fillna(0)
    n = target_data.iloc[0,0]
    y = target_data.iloc[1,0]
    # Calculate sample entropy
    sample_ent = entropy(y,n)

    # Determine the highest information gain (IG)
    best_attribute = ''
    high_gain = 0
    for name in col_names:
        list_branch = examples.groupby([str(name)])[str(target)]
        gain = gain_calc(list_branch, target, sample_ent, examples.shape[0], 0)
        if gain > high_gain:
            best_attribute = name
            high_gain = gain
    # col_names = col_names.remove(str(best_attribute))
    col_names.remove(best_attribute)
    # Establish root of tree with the highest IG
    tree = MyTree(examples, col_names, best_attribute, target)
    return tree


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

# # Evaluating the tree on the test data
# correct = 0
# for i in range(0,len(test)):
#     if str(tree.predicts(test.loc[i])) == str(test.loc[i,target]):
#         correct += 1
# print("\nThe accuracy is: ", correct/len(test))