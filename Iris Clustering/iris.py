# Kütüphaneler yüklenir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Veri kümesi yüklenir
iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None)

# Özellikler ve sınıf etiketleri ayrılır
X = iris.iloc[:, :-1].values
y = iris.iloc[:, -1].values

# Özellikler ölçeklendirilir
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dirsek yöntemi kullanılarak optimal küme sayısı belirlenir
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Dirsek Yöntemi')
plt.xlabel('Küme Sayısı')
plt.ylabel('WCSS')
plt.show()

# K-means algoritması ile kümeleme yapılır
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Kümeleme sonuçlarının görselleştirilmesi yapılır
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Setosa')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Versicolour')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Virginica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')
plt.title('Iris Veri Kümesi K-means Kümeleme Sonuçları')
plt.xlabel('Sepal Uzunluğu (cm)')
plt.ylabel('Sepal Genişliği (cm)')
plt.legend()
plt.show()
