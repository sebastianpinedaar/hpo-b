
import random
from hpob_handler import HPOBHandler
import matplotlib.pyplot as plt

class RandomSearch:

    def __init__(self):

        print("Using random search method...")

    def observe_and_suggest(self, X_obs, y_obs, X_pen):

        size_pending_eval = len(X_pen)
        idx = random.randint(0, size_pending_eval-1)
        return idx



if __name__ == "__main__":
    
    hpob_hdlr = HPOBHandler(root_dir="HPOB-Bench/", mode="test")
    rs_method = RandomSearch()
    search_space_id =  hpob_hdlr.get_search_spaces()[0]
    dataset_id = hpob_hdlr.get_datasets(search_space_id)[0]
    perf_hist = hpob_hdlr.evaluate(rs_method, search_space_id = search_space_id, 
                                                dataset_id = dataset_id,
                                                trial = "test0",
                                                n_iterations = 100 )
    plt.plot(perf_hist)
    plt.show()

        
