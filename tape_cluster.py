import argparse
import json
import pickle

import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from tape import ProteinBertModel, TAPETokenizer
import matplotlib.pyplot as plt


def main(args):
    np.random.seed(args.seed)
    data_file = "data/{0}/{0}_{1}.json".format(args.name, args.suffix)
    with open(data_file) as f:
        data = json.load(f)

    model = ProteinBertModel.from_pretrained("bert-base")
    tokenizer = TAPETokenizer(vocab="iupac")

    outputs = []
    labels = []
    print("total_size:", len(data))
    for index, entry in enumerate(data):
        protein_seq = entry["primary"]
        tokenized_input = torch.tensor([tokenizer.encode(protein_seq)])
        curr_output = model(tokenized_input)[0].squeeze()
        outputs.append(curr_output.mean(dim=0).cpu().detach().numpy())
        if args.cluster_label != None:
            labels.append(entry[args.cluster_label])
        else:
            labels.append(0)
        if index % 1000 == 0:
            print("processed {0}".format(index))

    sequence_output = np.vstack(outputs)
    print("doing pca")
    pca_output = PCA(n_components=50).fit_transform(sequence_output)
    print("doing tsne")
    tsne_output = TSNE(n_components=2).fit_transform(pca_output)
    plot_data = {}
    plot_data["pca"] = pca_output
    # plot_data["data"] = data
    plot_data["x"] = tsne_output[:, 0]
    plot_data["y"] = tsne_output[:, 1]
    plot_data["label"] = labels

    with open("saved/{0}_{1}_{2}".format(args.name, args.suffix, args.cluster_label), "wb") as handle:
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
    plt.savefig("saved/{0}_{1}_{2}.png".format(args.name, args.suffix, args.cluster_label), dpi=400)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, default="remote_homology")
    ap.add_argument("--suffix", type=str, default="train")
    ap.add_argument("--cluster_label", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap = ap.parse_args()
    main(ap)
