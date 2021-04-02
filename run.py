import time

# Run the clustering algorithms on the datasets and describe what you see.
start = time.time()
from part1 import run
end = time.time()
print('part1 time taken: ', start - end)
# Apply the dimensionality reduction algorithms to the two datasets and describe what you see.
start = time.time()
from part2 import run
end = time.time()
print('part2 time taken: ', start - end)
# Reproduce your clustering experiments, but on the data after you've run dimensionality reduction on it. Yes, that’s 16 combinations of datasets, dimensionality reduction, and clustering method. You should look at all of them, but focus on the more interesting findings in your report.
start = time.time()
from part3 import run
end = time.time()
print('part3 time taken: ', start - end)
# Apply the dimensionality reduction algorithms to one of your datasets from assignment #1 (if you've reused the datasets from assignment #1 to do experiments 1-3 above then you've already done this) and rerun your neural network learner on the newly projected data.
start = time.time()
from part4 import run
end = time.time()
print('part4 time taken: ', start - end)
# Apply the clustering algorithms to the same dataset to which you just applied the dimensionality reduction algorithms (you've probably already done this), treating the clusters as if they were new features. In other words, treat the clustering algorithms as if they were dimensionality reduction algorithms. Again, rerun your neural network learner on the newly projected data.
start = time.time()
from part5 import run
end = time.time()
print('part5 time taken: ', start - end)
