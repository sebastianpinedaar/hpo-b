
import random
from hpob_handler import HPOBHandler
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood


class RandomSearch:

    def __init__(self):

        print("Using random search method...")

    def observe_and_suggest(self, X_obs, y_obs, X_pen):

        size_pending_eval = len(X_pen)
        idx = random.randint(0, size_pending_eval-1)
        return idx


class GaussianProcess:

    def __init__(self):

        print("Using Gaussian Process as method...")
        
        #initialize acquisition functions

    def observe_and_suggest(self, X_obs, y_obs, X_pen):

        #fit the gaussian process
        gp = SingleTaskGP(X_obs, y_obs)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)

        acq = UpperConfidenceBOund(gp, beta=0.1)
        eval_acq = acq(X_pen)
        print(eval_acq)
        #sample the acquisition functions



if __name__ == "__main__":
    
    hpob_hdlr = HPOBHandler(root_dir="HPOB-Bench/", mode="test")
    rs_method = RandomSearch()
    method = GaussianProcess()
    search_space_id =  hpob_hdlr.get_search_spaces()[0]
    dataset_id = hpob_hdlr.get_datasets(search_space_id)[0]
    perf_hist = hpob_hdlr.evaluate(method, search_space_id = search_space_id, 
                                                dataset_id = dataset_id,
                                                trial = "test0",
                                                n_iterations = 100 )
    plt.plot(perf_hist)
    plt.show()

        
