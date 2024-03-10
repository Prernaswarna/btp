# Import pandas library
import pandas as pd
from numpy import unique
from numpy import where
from sklearn.cluster import MeanShift
from matplotlib import pyplot
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.mixture import GaussianMixture
  
# initialize list of lists
data = [[0.9198772, 0.8401239, 0.7996389], [0.8178693, 0.5104186, 0.4028717], [0.8852428, 0.836827, 0.8134385], [0.7427259, 0.5543912, 0.4160152], [0.4744705, 0.6636731, 0.2792631], [0.9323888, 0.8894005, 0.8682813], [0.7358057, 0.5530862, 0.3895114], [0.6147435, 0.7011861, 0.4652051], [0.7545342, 0.7817095, 0.6302957], [0.47935, 0.6537217, 0.3373841], [0.5629842, 0.57975, 0.2469351], [0.6343374, 0.680985, 0.504757], [0.8369481, 0.7676022, 0.6269168], [0.8351108, 0.7075708, 0.6063256], [0.5669668, 0.562534, 0.2980432], [0.9244207, 0.8840249, 0.8727653], [0.8905491, 0.783396, 0.6958309], [0.4607291, 0.5166897, 0.2043089], [0.5378487, 0.4940737, 0.2389557], [0.5846773, 0.440981, 0.2819954]]
  
# Create the pandas DataFrame
df = pd.DataFrame(data, columns=['Q1', 'Q2', 'Q3'])
  
# print dataframe.
df

print("Start of cell")
model = KMeans(n_clusters=4)
# model = MeanShift()
# model = DBSCAN(eps=0.15, min_samples=3)
# model = MiniBatchKMeans(n_clusters=5)
# model = Birch(threshold=0.1, n_clusters=4)
# model = AgglomerativeClustering(n_clusters=5)
# model = AffinityPropagation(damping=0.5)
# model = GaussianMixture(n_components=4)
# fit model and predict clusters
print("model defined")
yhat = model.fit_predict(df)
print("model done")
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
# for cluster in clusters:
#     print(cluster)
#  # get row indexes for samples with this cluster
#  row_ix = where(yhat == cluster)
#  # create scatter of these samples

# #  pyplot.scatter(df[row_ix, 0], df[row_ix, 1])
# # # show the plot
# # pyplot.show()

for i in yhat:
    print(i)
