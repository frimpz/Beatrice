import math
import torch
import pandas as pd
from sklearn import linear_model
from sklearn.cluster import KMeans
import scipy.stats as stats
import pylab
from kneed import KneeLocator
import pickle
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


def k_meansElbow(data, type):
    Sum_of_squared_distances = []
    _min = 3
    _max = 40
    stp = 5
    K = range(_min, _max)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(data)
        Sum_of_squared_distances.append(km.inertia_)
    kn = KneeLocator(K, Sum_of_squared_distances, curve='convex', direction='decreasing')
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k --- ' + type)
    plt.text(_max - 2 * stp, max(Sum_of_squared_distances), 'k = %d' % kn.knee)
    plt.show()


def plot_optimal_clusters(data, type):
    k_meansElbow(data, type)
    exit()


# Cluster with K-means
# K-Means
def k_means(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans_model = kmeans.fit(data)
    # pickle.dump(kmeans_model, open("kmeans_model.pkl", 'wb'))
    kmeans_labels = kmeans.labels_
    return kmeans_labels


def try_data(_prediction):
    final = torch.sigmoid(_prediction)
    return final


def cal_outlier(x, q_1, q_3):
    if x < q_1:
        return False
    elif x > q_3:
        return False
    else:
        return True


# remove outliers
def rem_out(xx, yy):
    y_q1 = np.quantile(yy, .25)
    y_q3 = np.quantile(yy, .75)
    y_iqr = y_q3 - y_q1
    y_q1 = y_q1 - 1.5*y_iqr
    y_q3 = y_q3 + 1.5*y_iqr
    xxx = []
    yyy =[]
    for i, j in zip(xx, yy):
        y = cal_outlier(j, y_q1, y_q3)
        if y:
            xxx.append(i)
            yyy.append(j)
    return np.array(xxx), np.array(yyy), ' -- No Outlier'


def pickle_save(filename, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def pickle_load(filename):
    with open(filename, "rb") as f:
        x = pickle.load(f)
        return x


data_df = pd.read_pickle("saved_hs")

clusters = range(0, 30)


for _cluster in clusters:

    cluster = [_cluster]

    filtered_df = data_df[data_df['KMeans-clusters'].isin(cluster)]
    filtered_df = filtered_df[filtered_df['scores'].notnull()]
    filtered_df = filtered_df.drop(columns=['gensets', 'KMeans-clusters'])

    _data = filtered_df.to_numpy()
    X = _data[:, : -1]
    y = _data[:, -1]

    # # remove outliers
    # X, y, tmp = rem_out(X, y)
    # title = title + tmp

    rs = ShuffleSplit(test_size=0.10)
    train_index, test_index = list(rs.split(X))[0]

    x_train = X[train_index]
    y_train = y[train_index]

    x_test = X[test_index]
    y_test = y[test_index]

    # create new model
    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)

    y_pred_test = model.predict(x_test)
    print('Mean squared error linear reg: %.2f'% mean_squared_error(y_test, y_pred_test))
    print('Coefficient of determination linear reg: %.2f'% r2_score(y_test, y_pred_test))

    y_pred_all = model.predict(X)

    linreg = stats.linregress(y, y_pred_all)

    fig, ax = plt.subplots()
    plt.scatter(y, y_pred_all, color='black')
    # temp = [linreg.intercept + linreg.slope.item() * i for i in y]
    # plt.plot(y, temp, 'r')

    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    ax.legend([extra, extra], ('R2 = %0.2f' % linreg.rvalue, 'Slope = %0.2f' % linreg.slope))

    # Just for graph title
    title = 'MM-GO Cluster: ' + str(cluster) + ' : ' + " Outliers"
    # tmp = ' -- All'

    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.savefig("comb-plots/" + str(_cluster) + "a", dpi=100)
    plt.show()
    plt.clf()

    node_count = len(y_pred_all)

    for i in range(len(y_pred_all)):
        if y_pred_all[i] < 0:
            y_pred_all[i] = 0

    y_res = y - y_pred_all
    y_sse = (y - y_pred_all) ** 2
    SD = math.sqrt(sum(y_sse)/(len(y_sse)-2))

    _y = []
    _x = []
    for i in zip(X, y, y_res):
        if abs(i[2]) < 2*SD:
            _x.append(i[0])
            _y.append(i[1])

    X = np.array(_x)
    y = np.array(_y)

    model.fit(X, y)
    y_pred_all = model.predict(X)

    drop_count = len(y_pred_all)

    # scatter plot section
    linreg = stats.linregress(y, y_pred_all)

    fig, ax = plt.subplots()
    plt.scatter(y, y_pred_all,  color='black')
    # temp = [linreg.intercept + linreg.slope.item()*i for i in y]
    # plt.plot(y, temp, 'r')

    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    ax.legend([extra, extra, extra, extra], ('R2 = %0.2f' % linreg.rvalue,
                                             'Slope = %0.2f' % linreg.slope,
              'Nodes = %d' % node_count, 'Ret = %d' % drop_count))

    # Just for graph title
    title = 'MM-GO Cluster: ' + str(cluster) + ' : ' + " No Outliers"
    # tmp = ' -- All'

    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.savefig("comb-plots/" + str(_cluster) + "b", dpi=100)
    plt.show()
    plt.clf()

    title = "Normal Quantile plot -- MM-GO" + str(cluster)
    c = stats.probplot(y_res, dist='norm', plot=pylab)
    plt.title(title)
    plt.savefig("comb-plots/" + str(_cluster) + "c", dpi=100)
    pylab.show()
    plt.clf()
