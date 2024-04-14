import math


class Node:
    def __init__(self):
        self.value = None  # Feature name or possible choice (feature value)
        self.next = None  # Next node
        self.childs = None  # Branches with possible choices


class DecisionTreeClassifier:

    def __init__(self, predictors_X, feature_names, labels):
        self.X = predictors_X  # features or predictors
        self.feature_names = feature_names
        self.labels = labels  # categories
        self.labelCategories = list(set(labels))  # unique categories
        self.labelCategoriesCount = [list(labels).count(x) for x in
                                     self.labelCategories]  # number of instances of each category
        self.node = None
        self.entropy = self._get_entropy([x for x in range(len(self.labels))])  # initial entropy of the system

    def _get_entropy(self, x_ids):
        labels = [self.labels[i] for i in x_ids]  # labels by instance id sorted
        label_count = [labels.count(x) for x in self.labelCategories]  # of instances each category

        entropy = sum([-count / len(x_ids) * math.log(count / len(x_ids), 2)
                       if count else 0
                       for count in label_count
                       ])

        return entropy

    def _get_information_gain(self, x_ids, feature_id):
        info_gain = self._get_entropy(x_ids)
        x_features = [self.X[x][feature_id] for x in x_ids]  # values of the chosen feature
        feature_vals = list(set(x_features))  # unique values
        feature_v_count = [x_features.count(x) for x in feature_vals]
        feature_v_id = [
            [x_ids[i]
             for i, x in enumerate(x_features)
             if x == y]
            for y in feature_vals
        ]

        info_gain_feature = sum([v_counts / len(x_ids) * self._get_entropy(v_ids)
                                 for v_counts, v_ids in zip(feature_v_count, feature_v_id)])

        info_gain = info_gain - info_gain_feature

        return info_gain

    def _get_feature_max_information_gain(self, x_ids, feature_ids):
        features_entropy = [self._get_information_gain(x_ids, feature_id) for feature_id in
                            feature_ids]  # entropy for each feature
        max_id = feature_ids[features_entropy.index(max(features_entropy))]  # find feat max info gain

        return self.feature_names[max_id], max_id

    def id3(self):
        x_ids = [x for x in range(len(self.X))]  # id assign instance
        feature_ids = [x for x in range(len(self.feature_names))]  # id assign faeture
        self.node = self._id3_recv(x_ids, feature_ids, self.node)

    def _id3_recv(self, x_ids, feature_ids, node):
        if not node:
            node = Node()
        labels_in_features = [self.labels[x] for x in x_ids]  # sort labels by instance id
        if len(set(labels_in_features)) == 1:  # for all surviced or not surviced return 1 or 0
            node.value = self.labels[x_ids[0]]
            return node
        if len(feature_ids) == 0:  # for no more features left return most probable label
            node.value = max(set(labels_in_features), key=labels_in_features.count)
            return node

        best_feature_name, best_feature_id = self._get_feature_max_information_gain(x_ids, feature_ids)
        node.value = best_feature_name
        node.childs = []
        feature_values = list(set([self.X[x][best_feature_id] for x in x_ids]))

        for value in feature_values:
            child = Node()
            child.value = value  # add feature choice as value or print
            node.childs.append(child)  # append new child (feature choice) to current node
            child_x_ids = [x for x in x_ids if self.X[x][best_feature_id] == value]
            if not child_x_ids:
                child.next = max(set(labels_in_features), key=labels_in_features.count)
                print('')
            else:
                if feature_ids and best_feature_id in feature_ids:
                    to_remove = feature_ids.index(best_feature_id)
                    feature_ids.pop(to_remove)
                child.next = self._id3_recv(child_x_ids, feature_ids, child.next)  # recursion
        return node

    def predict(self, ):
        return "survived"