import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.model_selection import train_test_split
#from sklearn import metrics

iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)

x = iris_df.iloc[:, [0, 1, 2, 3]].values

from sklearn.cluster import KMeans
kmn = [] 

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter = 300, n_init = 10,random_state=0)
    kmeans.fit(x)
    kmn.append(kmeans.inertia_)
    
'Plotting the graph'
plt.plot(range(1,11), kmn)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()

'Applying kmeans to the dataset / Creating the kmeans classifier'
kmeans = KMeans(n_clusters=3,init='k-means++',max_iter =300,n_init=10,random_state=0)
y_kmeans = kmeans.fit_predict(x)

'Visualising the clusters - On the first two columns'
plt.scatter(x[y_kmeans==0,0], x[y_kmeans== 0,1],s =100, c ='red', label='Setosa')
plt.scatter(x[y_kmeans==1,0], x[y_kmeans== 1,1],s =100, c ='blue', label='Versicolour')
plt.scatter(x[y_kmeans==2,0], x[y_kmeans== 2,1],s =100, c ='green', label='Virginica')

'Plotting the centroids of the clusters'
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1],s = 100, c = 'yellow', label = 'Centroids')
plt.legend()