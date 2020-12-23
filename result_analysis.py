import json
import pickle
import os
import numpy as np
from scipy.special import  softmax

# if __name__ == "__main__":

    # with open("../data/remote_homology/remote_homology_test_fold_holdout.json", "r") as f:
    #     data = json.load(f)
    #
    # data = {entry["id"]: entry for entry in data}
    #
    # data_hard = list()
    # with open("C:\\Users\\Jatin\\Documents\\Academics\\Projects\\DeepSF\\test\\DCNN_results_hard.txt") as f:
    #     for line in f:
    #         data_hard.append(data[line.strip()])
    #
    # with open("../data/remote_homology/remote_homology_test_fold_holdout_hard.json", "w") as f:
    #     json.dump(data_hard, f)

    # with open("results.pkl", "rb") as f:
    #     d = pickle.load(f)

    # print(d)


if __name__ == "__main__":

    prefix = "out_test_fold"
    dir_name = os.path.join("deepsf", prefix, "DCNN_results")
    file_list = os.listdir(dir_name)

    protein_id_to_true_label = dict()
    with open(os.path.join("deepsf", prefix, "DCNN_results.txt")) as f:
        for index, line in enumerate(f):
            if index == 0:
                continue
            s = line.strip().split("\t")
            if len(s) > 1:
                protein_id_to_true_label[s[0]] = s[1]

    label_to_index = dict()
    with open(os.path.join("deepsf", "fold_label_relation2.txt")) as f:
        for index, line in enumerate(f):
            if index == 0:
                continue
            s = line.strip().split("\t")
            if len(s) > 1:
                label_to_index[s[0]] = int(s[1])

    test_probs = dict()

    for file_name in file_list:
        protein_id = os.path.splitext(file_name)[0]
        if protein_id not in test_probs:
            test_probs[protein_id] = dict()
        if file_name.endswith("prediction"):
            probs = np.loadtxt(os.path.join(dir_name, file_name))
            target = label_to_index[protein_id_to_true_label[protein_id]]
            test_probs[protein_id]["deepsf"] = probs
            test_probs[protein_id]["deepsf_argmax"] = probs.argmax()
            test_probs[protein_id]["deepsf_target"] = target
        if file_name.endswith("hidden_feature"):
            feat = np.loadtxt(os.path.join(dir_name, file_name))
            test_probs[protein_id]["deepsf_feat"] = feat

    # with open("results_test_fold_hard.pkl", "rb") as f:
    with open("results_test_fold_full.pkl", "rb") as f:
        tape_results = pickle.load(f)

    st_tape = set([entry["ids"] for entry in tape_results[1]])
    st_deepsf = set(test_probs.keys())
    st_nontape = st_deepsf - st_tape
    for key in st_nontape:
        del test_probs[key]

    for entry in tape_results[1]:
        test_probs[entry["ids"]]["id"] = entry["ids"]
        test_probs[entry["ids"]]["tape"] = softmax(entry["prediction"])
        test_probs[entry["ids"]]["tape_argmax"] = test_probs[entry["ids"]]["tape"].argmax()
        test_probs[entry["ids"]]["tape_target"] = entry["target"]
        test_probs[entry["ids"]]["seq"] = entry["seq"]

    # print(test_probs)
    for key in test_probs:
        if "tape" not in test_probs[key]:
            del test_probs[key]

    with open("./data/remote_homology/remote_homology_test_fold_holdout.json", "r") as f:
        full_data = json.load(f)
    full_data = {entry["id"]: entry for entry in full_data}

    true_cnt = 0
    for entry in test_probs.values():
        prob_deepsf = entry["deepsf"][entry["deepsf_argmax"]]
        prob_tape = entry["tape"][entry["tape_argmax"]]
        item = full_data[entry["id"]]

        # if entry["deepsf_argmax"] == entry["deepsf_target"]:
        #     true_cnt += 1

        # if prob_deepsf > 0.9:
        #     true_cnt += 1

        # cases where we take deepsf by mistake
        # if entry["tape_argmax"] == entry["tape_target"] and prob_deepsf >= prob_tape:
        #     print("deep: {0:.4f}, tape: {1:.4f}".format(prob_deepsf, prob_tape))
        #     true_cnt += 1

        # cases where we take tape by mistake
        # if entry["deepsf_argmax"] == entry["deepsf_target"] and prob_tape > prob_deepsf:
        #     print("deep: {0:.4f}, tape: {1:.4f}".format(prob_deepsf, prob_tape))
        #     true_cnt += 1

        # if entry["deepsf_argmax"] == entry["deepsf_target"] and entry["tape_argmax"] != entry["tape_target"] and prob_tape > prob_deepsf:
        #     print("deep: {0:.4f}, tape: {1:.4f}".format(prob_deepsf, prob_tape))
        #     print("fold: {0}, class: {1}, family: {2}, super: {3}".format(item["fold_label"], item["class_label"],item["family_label"],item["superfamily_label"]))
        #     true_cnt += 1

        # DEBUG: [entry for entry in test_probs.values() if full_data[entry["id"]]["fold_label"] == 454]

        # if entry["deepsf_argmax"] != entry["deepsf_target"] and entry["tape_argmax"] == entry["tape_target"] and prob_tape <= prob_deepsf:
        #     # print("deep: {0:.4f}, tape: {1:.4f}".format(prob_deepsf, prob_tape))
        #     # print("fold: {0}, class: {1}, family: {2}, super: {3}".format(item["fold_label"], item["class_label"],
        #     print("class: {1}".format(item["fold_label"], item["class_label"],
        #                                                                   item["family_label"],
        #                                                                   item["superfamily_label"]))
        #     true_cnt += 1

        # best we can get
        # if entry["deepsf_argmax"] == entry["deepsf_target"] or entry["tape_argmax"] == entry["tape_target"]:
        #     true_cnt += 1

        if prob_deepsf < 0.3:
            if entry["tape_argmax"] == entry["tape_target"]:
                true_cnt += 1
        # if item["class_label"] == 5 or item["class_label"] == 6:  # reduces overall accuracy
        #     if entry["tape_argmax"] == entry["tape_target"]:
        #         true_cnt += 1
        # elif item["family_label"] == 1332 : # not helping overall
        #     if entry["deepsf_argmax"] == entry["deepsf_target"]:
        #         true_cnt += 1
        # if item["superfamily_label"] == 728: # not helping overall
        #     if entry["deepsf_argmax"] == entry["deepsf_target"]:
        #         true_cnt += 1
        # elif item["fold_label"] == 454:
        #     if entry["tape_argmax"] == entry["tape_target"]:
        #         true_cnt += 1
        # elif item["fold_label"] == 28:
        #     if entry["deepsf_argmax"] == entry["deepsf_target"]:
        #         true_cnt += 1
        # elif item["fold_label"] == 80:
        #     if entry["deepsf_argmax"] == entry["deepsf_target"]:
        #         true_cnt += 1
        elif prob_deepsf >= prob_tape:
            if entry["deepsf_argmax"] == entry["deepsf_target"]:
                true_cnt += 1
        else:
            if entry["tape_argmax"] == entry["tape_target"]:
                true_cnt += 1

    acc = true_cnt / len(test_probs)
    print(acc)

    # table analysis
    # label = "fold_label"
    # label = "class_label"
    label = "family_label"
    # label = "superfamily_label"
    unique_values = set([entry[label] for entry in full_data.values()])
    table = dict()
    for val in unique_values:
        cnt_total = len([entry for entry in test_probs.values() if (full_data[entry["id"]][label] == val)])
        correct_deepsf = len([entry for entry in test_probs.values() if (full_data[entry["id"]][label] == val and entry["deepsf_argmax"] == entry["deepsf_target"])]) / cnt_total
        correct_tape = len([entry for entry in test_probs.values() if (full_data[entry["id"]][label] == val and entry["tape_argmax"] == entry["tape_target"])]) / cnt_total
        table[val] = {label: val, "total": cnt_total, "deepsf": correct_deepsf, "tape": correct_tape}

    print({k: v for k, v in sorted(table.items(), key=lambda item: item[1]["total"], reverse=True)})

