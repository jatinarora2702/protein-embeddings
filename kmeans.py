import pickle
import argparse
from sklearn.cluster import KMeans
from sklearn import metrics
from collections import defaultdict
import numpy as np


def main(args):
    with open("saved/{0}_{1}_{2}".format(args.name, args.suffix, args.cluster_label), "rb") as handle:
        plot_data = pickle.load(handle)
    orig = defaultdict(int)
    for l in plot_data["label"]:
        orig[l] += 1
    print(dict(sorted(orig.items(), key=lambda item: item[1])))
    print("training kmeans")
    # tsne_data = np.hstack([np.array(plot_data["x"]).reshape(-1, 1), np.array(plot_data["y"]).reshape(-1, 1)])
    kmeans = KMeans(len(orig)).fit(plot_data["pca"])
    # kmeans = KMeans(len(orig)).fit(tsne_data)
    print("predicting")
    outputs = kmeans.predict(plot_data["pca"])
    # outputs = kmeans.predict(tsne_data)
    d = defaultdict(int)
    for l in outputs:
        d[l] += 1
    print(dict(sorted(d.items(), key=lambda item: item[1])))
    print(metrics.adjusted_rand_score(plot_data["label"], outputs))
    
    lst1 = sorted(orig.items(), key=lambda item: item[1])
    lst2 = sorted(d.items(), key=lambda item: item[1])
    mapping = dict()
    for i in range(len(lst2)):
        mapping[lst2[i][0]] = lst1[i][0]
    print(mapping)
    print(metrics.accuracy_score(plot_data["label"], [mapping[k] for k in outputs]))



if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--name", type=str, default="remote_homology")
	ap.add_argument("--suffix", type=str, default="train")
	ap.add_argument("--cluster_label", type=str, default=None)
	ap = ap.parse_args()
	main(ap)
