# Homework 2
# Name: Dean Kelley
# Professor: Martine De Cock
# description: Training and testing decision trees with discrete-values attributes

import math
import pandas as pd


class DecisionNode:

    # A DecisionNode contains an attribute and a dictionary of children.
    # The attribute is either the attribute being split on, or the predicted label if the node has no children.
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = {}

    # Visualizes the tree
    def display(self, level=0):
        if self.children == {}:  # reached leaf level
            print(": ", self.attribute, end="")
        else:
            for value in self.children.keys():
                prefix = "\n" + " " * level * 4
                print(prefix, self.attribute, "=", value, end="")
                self.children[value].display(level + 1)

    # Predicts the target label for instance x
    def predicts(self, x):
        if self.children == {}:  # reached leaf level
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


# Calculates entropy from yes/no counts
def entropy(y, n):
    s = n + y
    if y == 0:
        y_log = 0
    else:
        y_log = y / s * math.log2(y / s)
    if n == 0:
        n_log = 0
    else:
        n_log = n / s * math.log2(n / s)
    ent = -(y_log + n_log)
    return ent


# group is examples to calculate gain from
# target is target attribute
# ent_prior is branch entropy
# s_tot is sample size of group
# flag is a boolean value to tell program if it's calculating entire sample or branch sample
def gain_calc(group, goal, ent_prior, s_tot, flag):
    counts = group
    if flag:  # Information gain of attribute from total example sample
        counts = counts.value_counts().to_frame(name='count').reset_index()
        counts = counts.pivot_table(index=str(goal), columns=counts.columns[0], values='count').fillna(0)
        ent = 0
        for column in counts:
            n = counts[column].values[0]
            y = counts[column].values[1]
            s = y + n
            ent_branch = s / s_tot * entropy(y, n)
            ent = ent + ent_branch

        gain = ent_prior - ent
    else:  # Information gain of attribute from branch condition
        counts = counts.pivot_table(index=str(goal), columns=counts.columns[0], values='count').fillna(0)
        ent = 0
        ent_branch = 0
        for column in counts:
            n = counts[column].values[0]
            y = counts[column].values[1]
            s = y + n
            ent_branch = s / s_tot * entropy(y, n)
            ent = ent + ent_branch
        gain = ent_prior - ent
    return gain


def tree_builder(examples, attributes, branch, goal):
    conditions = attributes
    my_tree = DecisionNode(str(branch))
    branch_bool = examples.groupby([str(branch)])[str(goal)].value_counts().to_frame(name='count').reset_index()
    branch_bool = branch_bool.pivot_table(index=str(goal), columns=branch_bool.columns[0], values='count').fillna(0)
    # calculate branch entropy
    new_branch = ''
    for column in branch_bool:  # Determine next branch
        n = branch_bool[column].values[0]
        y = branch_bool[column].values[1]
        ent = entropy(y, n)
        if ent == 0 or len(conditions) == 0:  # Makes a decision
            leaf = ''
            if y > n:
                leaf = 'yes'
            else:
                leaf = 'no'
            my_tree.children[column] = DecisionNode(leaf)
        else:  # Needs more info
            max_gain = 0
            for name in conditions:
                branch_group = examples.groupby([str(branch), str(name)])[str(goal)].value_counts().to_frame(
                    name='count').reset_index()
                branch_group = branch_group.astype({str(branch): 'string'})
                branch_group = branch_group[branch_group[str(branch)].str.contains(str(column)) == True]
                branch_group = branch_group.drop(str(branch), axis=1)
                s_tot = branch_group['count'].sum()
                gain = gain_calc(branch_group, goal, ent, s_tot, False)
                if gain >= max_gain:
                    new_branch = name
                    max_gain = gain
            if len(conditions) != 0:
                conditions.remove(str(new_branch))
                next_branch = tree_builder(examples, conditions, new_branch, goal)
                my_tree.children[column] = next_branch
    return my_tree


# Begin ID3 algorithm
def id3(examples, goal, attributes):
    conditions = attributes
    target_data = examples[str(goal)].value_counts().to_frame(name='count').reset_index()
    target_data = target_data.pivot_table(index=str(goal), columns=target_data.columns[0], values='count').fillna(0)
    n = target_data.iloc[0, 0]
    y = target_data.iloc[1, 0]
    # Calculate sample entropy
    sample_ent = entropy(y, n)
    # Determine the highest information gain (IG)
    best_attribute = ''
    high_gain = 0
    for name in conditions:
        list_branch = examples.groupby([str(name)])[str(goal)]
        gain = gain_calc(list_branch, goal, sample_ent, examples.shape[0], True)
        if gain > high_gain:
            best_attribute = name
            high_gain = gain
    conditions.remove(best_attribute)
    # Establish root of tree with the highest IG
    my_tree = tree_builder(examples, conditions, best_attribute, goal)
    return my_tree


####################   MAIN PROGRAM ######################

# Reading input data
choose = 'r'
if choose == 'p':
    train = pd.read_csv('playtennis_train.csv')
    test = pd.read_csv('playtennis_test.csv')
    target = 'playtennis'
else:
    train = pd.read_csv('republican_train.csv')
    test = pd.read_csv('republican_test.csv')
    target = 'republican'
my_attributes = train.columns.tolist()
my_attributes.remove(target)

# Learning and visualizing the tree
tree = id3(train, target, my_attributes)
tree.display()
# Evaluating the tree on the test data
correct = 0
for i in range(0, len(test)):
    if str(tree.predicts(test.loc[i])) == str(test.loc[i, target]):
        correct += 1
print("\nThe accuracy is: ", correct / len(test))
