#Before running this notebook download the data from https://rewind.tf.uni-freiburg.de/index.php/s/B6gY9cpZ65fBfGJ

import numpy as np
import pandas as pd
import json
import os


class BOSSHandler:

    def __init__(self, root_dir = "", mode = "test"):
        print("Loading boss handler")

        if mode == "test":
            self.meta_test_data, self.bo_initializations = self.load_data(root_dir, only_test=True)
        else:
            self.meta_train_data, self.meta_validation_data, 
            self.meta_test_data, self.bo_initializations = self.load_data(root_dir, only_test=False)

    def load_data(rootdir="", only_test = True):

        meta_train_path = os.path.join(rootdir, "meta-train-dataset.json")
        meta_test_path = os.path.join(rootdir,"meta-test-dataset.json")
        meta_validation_path = os.path.join(rootdir, "meta-validation-datset.json")
        bo_initializations_path = os.path.join(rootdir, "bo-initializations.json")

        with open(meta_test_path, "rb") as f:
            meta_test_data = json.load(f)
        
        with open(bo_initializations_path, "rb") as f:
            bo_initializations = json.load(f)

        if only_test:
            return meta_test_data, bo_initializations
        else:
            with open(meta_train_path, "rb") as f:
                meta_train_data = json.load(f)
            with open(meta_validation_path, "rb") as f:
                meta_validation_data = json.load(f)

        return meta_train_data, meta_validation_data, meta_test_data, bo_initializations

    def normalized_regret(self, y):

        return (y-np.min(y))/(np.max(y)-np.min(y))

    def evaluate (self, bo_method = None, search_space_id = None, dataset_id = None, trial = None, n_iterations = 10):

        assert bo_method!=None, "Provide a valid method object for evaluation."
        assert hasattr(bo_method, "observe_and_suggest"), "The provided  object does not have a method called ´observe_and_suggest´"
        assert search_space_id!= None, "Provide a valid search space id. See documentatio for valid obptions."
        assert dataset_id!= None, "Provide a valid dataset_id. See documentation for valid options."
        assert trial!=None, "Provide a valid initialization. Valid options are: test0, test1, test2, test3, test4."

        n_initial_evaluations = 5
        X = np.array(self.meta_dataset_test[search_space_id][dataset_id]["X"])
        y = np.array(self.meta_dataset_test[search_space_id][dataset_id]["y"])
        y = self.normalized_regret(y)
        data_size = len(X)
        
        pending_evaluations = list(range(data_size))
        current_evaluations = []        

        init_ids = self.bo_initializations[search_space_id][dataset_id][trial]
        
        for i in range(n_initial_evaluations):
            idx = init_ids[i]
            pending_evaluations.remove(idx)
            current_evaluations.append(idx)

        max_performance_history = []
        for i in range(n_iterations):

            idx = bo_method.observe_and_suggest(X[current_evaluations], y[current_evaluations], X[pending_evaluations])
            pending_evaluations.remove(idx)
            current_evaluations.append(idx)
            max_performance_history.append(np.max(y[current_evaluations]))
        
        return max_performance_history