# HPO-B
HPO-B is a benchmark for assessing the performance of HPO algorithms.

## Usage


```python
from hpob_handler import HPOBHandler
from bo_methods import RandomSearch
import matplotlib.pyplot as plt

hpob_hdlr = HPOBHandler(root_dir="HPO-Bench/", mode="test")

search_space_id =  hpob_hdlr.get_search_spaces()[0]
dataset_id = hpob_hdlr.get_datasets(search_space_id)[1]

method = RandomSearch()
perf = hpob_hdlr.evaluate(method, search_space_id = search_space_id, 
                                        dataset_id = dataset_id,
                                        trial = trial,
                                        n_iterations = 100 )


plt.plot(perf)


