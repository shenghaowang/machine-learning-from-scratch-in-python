import numpy as np
import os
import json
import operator
from operator import itemgetter

class MyDecisionTreeRegressor():
    def __init__(self, max_depth=5, min_samples_split=1):
        '''
        Initialization
        :param max_depth: type: integer
        maximum depth of the regression tree. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.
        :param min_samples_split: type: integer
        minimum number of samples required to split an internal node:

        root: type: dictionary, the root node of the regression tree.
        '''

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    # Combine X, y
    # Split a sample based on an attribute and threshold
    def _split_sample(self, attr_id, threshold, sample):
        left, right = list(), list()
        for row in sample:
            if row[attr_id] <= threshold:
                left.append(row)
            else:
                right.append(row)
        return np.array(left), np.array(right)

    # Calculate RSS for split dataset
    def _compute_rss(self, branches):
        left_branch, right_branch = branches
        left_rss = right_rss = 0
        if left_branch.any():
            left_ys = [row[-1] for row in left_branch]
            left_avgy = sum(left_ys) / len(left_ys)
            left_rss = sum([(left_avgy - y) ** 2 for y in left_ys])
        if right_branch.any():
            right_ys = [row[-1] for row in right_branch]
            right_avgy = sum(right_ys) / len(right_ys)
            right_rss = sum([(right_avgy - y) ** 2 for y in right_ys])
        return left_rss + right_rss

    def _get_cut_point(self, dataset):
        ys = [row[-1] for row in dataset]
        n_feature = dataset.shape[1] - 1
        min_rss = float('inf')
        opt_attr = 0
        opt_cut = 0
        opt_branches = None
        for i in range(n_feature):
            for row in dataset:
                branches = self._split_sample(i, row[i], dataset)
                rss = self._compute_rss(branches)
                if rss < min_rss:
                    opt_attr = i
                    opt_cut = row[i]
                    min_rss = rss
                    opt_branches = branches
        return {
            'splitting_variable': opt_attr,
            'splitting_threshold': opt_cut,
            'split_branches': opt_branches
        }


    def _compute_branch_value(self, branch):
        sum_y = 0
        for row in branch:
            sum_y += row[-1]
        return sum_y / len(branch)


    def _split(self, node, max_depth, min_samples_split, depth):
        left, right = node.get('split_branches')
        del(node['split_branches'])
        # Check for no split
        if not left.any() or not right.any():
            node['left'] = node['right'] = self._compute_node_value(left + right)
        # Check for max depth
        if depth >= max_depth:
            node['left'] = self._compute_branch_value(left)
            node['right'] = self._compute_branch_value(right)
            return
        # Process left child
        if len(left) < min_samples_split:
            node['left'] = self._compute_branch_value(left)
        else:
            node['left'] = self._get_cut_point(left)
            self._split(node['left'], max_depth, min_samples_split, depth+1)
        # Process right child
        if len(right) < min_samples_split:
            node['right'] = self._compute_branch_value(right)
        else:
            node['right'] = self._get_cut_point(right)
            self._split(node['right'], max_depth, min_samples_split, depth+1)


    def fit(self, X, y):
        '''
        Inputs:
        X: Train feature data, type: numpy array, shape: (N, num_feature)
        Y: Train label data, type: numpy array, shape: (N,)

        You should update the self.root in this function.
        '''

        N = X.shape[0]
        data = []
        for i in range(N):
            row = np.append(X[i], y[i])
            data.append(row)
        data = np.array(data)
        self.root = self._get_cut_point(data)
        self._split(self.root, self.max_depth, self.min_samples_split, 1)

        return self

    def predict(self, X):
        '''
        :param X: Feature data, type: numpy array, shape: (N, num_feature)
        :return: y_pred: Predicted label, type: numpy array, shape: (N,)
        '''
        def predict_sample(node, row):
            if row[node['splitting_variable']] <= node['splitting_threshold']:
                if isinstance(node['left'], dict):
                    return predict_sample(node['left'], row)
                else:
                    return node['left']
            else:
                if isinstance(node['right'], dict):
                    return predict_sample(node['right'], row)
                else:
                    return node['right']

        y_pred = []
        for row in X:
            label = predict_sample(self.root, row)
            y_pred.append(label)
        return np.array(y_pred)

    def get_model_string(self):
        model_dict = self.root
        return model_dict

    def save_model_to_json(self, file_name):
        model_dict = self.root
        with open(file_name, 'w') as fp:
            json.dump(model_dict, fp)


# For test
if __name__=='__main__':
    for i in range(3):
        x_train = np.genfromtxt("Test_data" + os.sep + "x_" + str(i) +".csv", delimiter=",")
        y_train = np.genfromtxt("Test_data" + os.sep + "y_" + str(i) +".csv", delimiter=",")

        for j in range(2):
            tree = MyDecisionTreeRegressor(max_depth=5, min_samples_split=j + 2)
            tree.fit(x_train, y_train)

            model_string = tree.get_model_string()
            # print(model_string)
            with open("Test_data" + os.sep + "decision_tree_" + str(i) + "_" + str(j) + ".json", 'r') as fp:
                test_model_string = json.load(fp)
            #print(test_model_string)
            print(operator.eq(model_string, test_model_string))
            # if not operator.eq(model_string, test_model_string):
            #     print(model_string)
            #     print(test_model_string)

            y_pred = tree.predict(x_train)

            y_test_pred = np.genfromtxt("Test_data" + os.sep + "y_pred_decision_tree_"  + str(i) + "_" + str(j) + ".csv", delimiter=",")
            print(np.square(y_pred - y_test_pred).mean() <= 10**-10)

