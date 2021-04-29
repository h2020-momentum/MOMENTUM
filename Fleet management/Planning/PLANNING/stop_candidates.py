from sklearn.cluster import KMeans
from math import radians, cos, sin, asin, sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km*1000

def get_cluster(df,n):
    x0 = df.iloc[:, :2]
    x0.columns = ['lon', 'lat']
    # print(x0.head())
    x1 = df.iloc[:,2:4]
    x1.columns = ['lon', 'lat']
    # print(x1.head())
    x = x0.append(x1).reset_index(drop = True)
    # print(x.head())

    kmeans = KMeans(n_clusters= n)
    kmeans.fit(x)
    kmeans.labels_
    

    x['label'] = kmeans.labels_
    return x , kmeans.cluster_centers_


def max_dist(df, centers):
    clst = df.label.iat[0]
    lon_c = [centers[clst,0] for _ in range(df.shape[0])]
    lat_c = [centers[clst,1] for _ in range(df.shape[0])]
    df['lon_c'] = lon_c
    df['lat_c'] = lat_c
    df['hdist'] =haversine_np(df['lon'], df['lat'],df['lon_c'],df['lat_c'])
    return df['hdist'].max()

def run_clusters(df,maxdist, Crange):
    rmse_list = {}
    for i in range(Crange[0], Crange[1] +1):
        x, centers  = get_cluster(df, i)
        rmse_in_cluster = np.sqrt(np.mean((x.groupby('label').apply(lambda x: max_dist(x,centers)).\
            reset_index(name= 'maxDistance')['maxDistance'] - maxdist)**2))

        mae_in_cluster = np.mean(abs(x.groupby('label').apply(lambda x: max_dist(x,centers)).\
            reset_index(name= 'maxDistance')['maxDistance'] - maxdist))

        rmse_list[i] =  [rmse_in_cluster, mae_in_cluster]
    return rmse_list


if __name__ == '__main__':
    data = pd.read_csv('sampe_od_data.csv')
    print(data.head())

    exp = run_clusters(data, 1000, [10,70])
    print(exp)
    pd.DataFrame.from_dict(exp , orient= 'index').plot()
    plt.show()