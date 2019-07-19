import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

df = pd.read_csv("market_segment.csv")
df.head()


#Plotting distributions of Quantity and Revenue to check skewness
sns.distplot(df['Quantity'])
sns.distplot(df['Revenue'])

#Log transformation to reduce the skewness
df['log_QTY'],df['log_rev'] = np.log(df['Quantity']), np.log(df['Revenue'])


sns.distplot(df['log_QTY'])
sns.distplot(df['log_rev'])

#Scaling and Zero centering - Standardization 

sc = StandardScaler()

a = df[["log_QTY","log_rev"]]
a = sc.fit_transform(a)
a = pd.DataFrame(a, columns = ["log_QTY","log_rev"])

sns.distplot(a["log_QTY"])
sns.distplot(a["log_rev"])

a.describe()
df.describe()


#Elbow Method

wcss = []
for i in range(1,11):
    km=KMeans(n_clusters=i,init='k-means++', max_iter=300, n_init=10, random_state=0)
    km.fit(a)
    wcss.append(km.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()



#Calculating Silhouetee score to find the optimal number of clusters

def optimal_kmeans(dataset, start=2, end=11):
    '''
    Calculate the optimal number of kmeans
    
    INPUT:
        dataset : dataframe. Dataset for k-means to fit
        start : int. Starting range of kmeans to test
        end : int. Ending range of kmeans to test
    OUTPUT:
        Values and line plot of Silhouette Score.
    '''
    
    # Create empty lists to store values for plotting graphs
    n_clu = []
    km_ss = []

    # Create a for loop to find optimal n_clusters
    for n_clusters in range(start, end):

        # Create cluster labels
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(dataset)

        # Calcualte model performance
        silhouette_avg = round(silhouette_score(dataset, labels, random_state=1), 3)

        # Append score to lists
        km_ss.append(silhouette_avg)
        n_clu.append(n_clusters)

        print("No. Clusters: {}, Silhouette Score: {}, Change from Previous Cluster: {}".format(
            n_clusters, 
            silhouette_avg, 
            (km_ss[n_clusters - start] - km_ss[n_clusters - start - 1]).round(3)))

        # Plot graph at the end of loop
        if n_clusters == end - 1:
            plt.figure(figsize=(6.47,3))

            plt.title('Silhouette Score')
            sns.pointplot(x=n_clu, y=km_ss)
            plt.savefig('silhouette_score.png', format='png', dpi=1000)
            plt.tight_layout()
            plt.show()
            
            
            
optimal_kmeans(a)        
            
#Kmeans Clustering

def kmeans(df, clusters_number):
    '''
    Implement k-means clustering on dataset
    
    INPUT:
        dataset : dataframe. Dataset for k-means to fit.
        clusters_number : int. Number of clusters to form.
        end : int. Ending range of kmeans to test.
    OUTPUT:
        Cluster results and t-SNE visualisation of clusters.
    '''
    
    kmeans = KMeans(n_clusters = clusters_number, random_state = 1)
    kmeans.fit(df)

    # Extract cluster labels
    cluster_labels = kmeans.labels_
        
    # Create a cluster label column in original dataset
    df_new = df.assign(Cluster = cluster_labels)
    
    # Initialise TSNE
    model = TSNE(random_state=1)
    transformed = model.fit_transform(df)
    
    # Plot t-SNE
    plt.title('Flattened Graph of {} Clusters'.format(clusters_number))  
    fig, ax = plt.subplots(figsize=(12,6))
    sns.scatterplot(x=transformed[:,0], y=transformed[:,1], hue=cluster_labels, palette="Set1") #style=cluster_labels, palette="Set1")
    ax.set(ylabel="Revenue") 
    ax.set(xlabel="Quantity") 

    return df_new, cluster_labels


result = kmeans(a,5)

b = result[0]

df["Cluster"] = b["Cluster"]


#Hence, we have segmented the products into 5 clusters based on their purchase pattern.

