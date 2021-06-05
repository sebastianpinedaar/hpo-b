# HPO-B
HPO-B is a benchmark for assessing the performance of HPO algorithms.

## Usage


```python
from hpob_handler import HPOBHandler
from bo_methods import RandomSearch
import matplotlib.pyplot as plt


hpob_hdlr = HPOBHandler(root_dir="../BOSS-Bench/", mode="test")

search_space_id =  hpob_hdlr.get_search_spaces()[0]
dataset_id = hpob_hdlr.get_datasets(search_space_id)[1]
perf_list = []

for acq_name in valid_acquisitions:
    perf_per_method = []
    for trial in trials:
        print("Using ", acq_name, " as acquisition function...")
        method = GaussianProcess(acq_name=acq_name)
        perf = hpob_hdlr.evaluate(method, search_space_id = search_space_id, 
                                                dataset_id = dataset_id,
                                                trial = trial,
                                                n_iterations = 100 )
        perf_per_method.append(perf)

    plt.plot(np.array(perf_per_method).mean(axis=0))
plt.legend(valid_acquisitions)
plt.show()
plt.savefig("Results.png")

