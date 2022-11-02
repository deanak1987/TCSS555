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


# # Calculates entropy from yes/no counts
# # y is yes count
# # n is no count
# def entropy(y, n):
#     s = n + y
#     if y == 0:
#         y_log = 0
#     else:
#         y_log = y / s * math.log2(y / s)
#     if n == 0:
#         n_log = 0
#     else:
#         n_log = n / s * math.log2(n / s)
#     ent = -(y_log + n_log)
#     return ent
#
#
# # Calculate information Gain
# # group is examples to calculate gain from
# # target is target attribute
# # ent_prior is branch entropy
# # s_tot is sample size of group
# def gain_calc(group, goal, ent_prior, s_tot):
#     counts = group.pivot_table(index=goal, columns=group.columns[0], values='count').fillna(0)
#     ent = 0
#     for column in counts:
#         n = counts[column].values[0]
#         y = counts[column].values[1]
#         s = y + n
#         ent_branch = s / s_tot * entropy(y, n)
#         ent = ent + ent_branch
#     gain = ent_prior - ent
#     return gain
#
#
# # Continue to create each branch
# # examples is the data to train tree off of
# # my_attributes is the list of attributes to choose from for each branch
# # ent_prior is the entropy of the parent branch
# # goal is the decision for each example
# def tree_builder(examples, my_attributes, branch, goal):
#     my_tree = DecisionNode(branch)
#     branch_bool = examples.groupby([branch])[goal].value_counts().to_frame(name='count').reset_index()
#     branch_bool = branch_bool.pivot_table(index=goal, columns=branch_bool.columns[0], values='count').fillna(0)
#     yes_no = branch_bool.axes[0].tolist()  # Get yes/no values
#     # Calculate branch entropy then calculate max gain to determine next branch
#     new_branch = ''
#     for column in branch_bool:  # Determine next branch
#         n = branch_bool[column].values[0]
#         y = branch_bool[column].values[1]
#         ent = entropy(y, n)
#         if ent == 0 or len(my_attributes) == 0:  # Makes a decision
#             if y > n:
#                 leaf = yes_no[1]
#             else:
#                 leaf = yes_no[0]
#             my_tree.children[column] = DecisionNode(leaf)
#         else:  # Needs more info
#             max_gain = 0
#             for name in my_attributes:
#                 branch_group = examples.groupby([branch, name])[goal].value_counts().to_frame(
#                     name='count').reset_index()
#                 branch_group = branch_group.astype({branch: 'string'})
#                 branch_group = branch_group[branch_group[branch].str.contains(str(column))].drop(branch, axis=1)
#                 s_tot = branch_group['count'].sum()
#                 gain = gain_calc(branch_group, goal, ent, s_tot)
#                 if gain >= max_gain:
#                     new_branch = name
#                     max_gain = gain
#             if len(my_attributes) != 0:  # More branches to build
#                 examples = examples.astype({str(branch): 'string'})
#                 new_examples = examples[examples[branch].str.contains(str(column))].drop(branch, axis=1)
#                 my_attributes.remove(new_branch)
#                 next_branch = tree_builder(new_examples, my_attributes, new_branch, goal)
#                 my_tree.children[column] = next_branch
#     return my_tree
#
#
# # Begin ID3 algorithm
# # examples is the data to train tree off of
# # my_attributes is the list of attributes to choose from for each branch
# # goal is the decision for each example
# def id3(examples, goal, my_attributes):
#     target_data = examples[goal].value_counts().to_frame(name='count').reset_index()
#     target_data = target_data.pivot_table(index=goal, columns=target_data.columns[0], values='count').fillna(0)
#     n = target_data.iloc[0, 0]
#     y = target_data.iloc[1, 0]
#     # Calculate sample entropy
#     sample_ent = entropy(y, n)
#     # Determine the highest information gain (IG)
#     best_attribute = ''
#     high_gain = 0
#     for name in my_attributes:
#         list_branch = examples.groupby([name])[goal].value_counts().to_frame(name='count').reset_index()
#         gain = gain_calc(list_branch, goal, sample_ent, examples.shape[0])
#         if gain > high_gain:
#             best_attribute = name
#             high_gain = gain
#     my_attributes.remove(best_attribute)
#     # Establish root of tree with the highest IG
#     my_tree = tree_builder(examples, my_attributes, best_attribute, goal)
#     return my_tree


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
attributes = train.columns.tolist()
attributes.remove(target)

# Learning and visualizing the tree
# tree = id3(train, target, attributes)
# tree.display()
# # Evaluating the tree on the test data
# correct = 0
# for i in range(0, len(test)):
#     if str(tree.predicts(test.loc[i])) == str(test.loc[i, target]):
#         correct += 1
# print("\nThe accuracy is: ", correct / len(test))
