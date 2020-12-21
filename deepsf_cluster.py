import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def main(args):
    prefix = "out_test_fold"
    dir_name = os.path.join(".", "deepsf", prefix, "DCNN_results")
    file_list = os.listdir(dir_name)

    protein_data = dict()

    for file_name in file_list:
        protein_id = os.path.splitext(file_name)[0]
        if protein_id not in protein_data:
            protein_data[protein_id] = dict()
        if file_name.endswith("hidden_feature"):
            feat = np.loadtxt(os.path.join(dir_name, file_name))
            protein_data[protein_id]["deepsf_feat"] = feat

    with open("results_test_fold_full.pkl", "rb") as f:
        tape_results = pickle.load(f)

    for entry in tape_results[1]:
        protein_data[entry["ids"]]["id"] = entry["ids"]
        protein_data[entry["ids"]]["seq"] = entry["seq"]

    with open("../data/remote_homology/remote_homology_test_fold_holdout.json", "r") as f:
        full_data = json.load(f)

    for entry in full_data:
        protein_data[entry["id"]]["class_label"] = entry["class_label"]
        protein_data[entry["id"]]["fold_label"] = entry["fold_label"]
        protein_data[entry["id"]]["family_label"] = entry["family_label"]
        protein_data[entry["id"]]["superfamily_label"] = entry["superfamily_label"]

    feats = np.array([entry["deepsf_feat"] for entry in protein_data.values()])
    labels = np.array([entry[args.cluster_label] for entry in protein_data.values()])

    pca_output = PCA(n_components=50).fit_transform(feats)
    print("doing tsne")
    tsne_output = TSNE(n_components=2).fit_transform(pca_output)
    plot_data = {}
    plot_data["pca"] = pca_output
    plot_data["x"] = tsne_output[:, 0]
    plot_data["y"] = tsne_output[:, 1]
    plot_data["label"] = labels

    with open("saved/deepsf_remote_homology_test_holdout_{0}".format(args.cluster_label), "wb") as handle:
        pickle.dump(plot_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("saved")

    plt.figure()
    ax = plt.subplot()
    sns.scatterplot(x="x", y="y", hue="label",
                    palette=sns.color_palette("hls", len(set(labels))),
                    data=plot_data,
                    legend="full",
                    alpha=0.3,
                    ax=ax)
    plt.savefig("saved/deepsf_remote_homology_test_holdout_{0}.png".format(args.cluster_label), dpi=400)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster_label", type=str, default="class_label")
    ap.add_argument("--seed", type=int, default=42)
    ap = ap.parse_args()
    main(ap)
