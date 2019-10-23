
dataset=[[1,2,3],[2,3,4],[4,5,6]]
import numpy as np
tmp=[0,1]
dataset = np.delete(dataset, tmp, axis=0)
print(dataset)

