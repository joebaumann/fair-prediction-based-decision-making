import pickle
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import utils

def load_compas_data(threshold_nr, sens_attr):
    print("\nStarting to load COMPAS data")
    filename = "data/propublica-recidivism_1_FOLD_preprocessed.pickle"
    with open(filename, 'rb') as f:
        my_dict_unpickled = pickle.load(f)
    my_dict_unpickled["privileged_val"] = "Caucasian"
    print("  COMPAS data loaded")

    group0_indices = my_dict_unpickled["train_df_sensitive_attrs"][sens_attr] != my_dict_unpickled["privileged_val"]
    group1_indices = my_dict_unpickled["train_df_sensitive_attrs"][sens_attr] == my_dict_unpickled["privileged_val"]
    group_indices = [group0_indices, group1_indices]

    print("  Groups have been defined: group_0 is NOT", str(
        my_dict_unpickled["privileged_val"]), "and group_1 is", str(my_dict_unpickled["privileged_val"]))

    s = my_dict_unpickled["predictions_proba_X_train"]
    y = my_dict_unpickled["y_train"]
    thresholds = utils.generate_thresholds(threshold_nr)
    y_pred_dict = {t: s.between(t[0], t[1]) for t in thresholds}

    group0_base_rate, group1_base_rate = mean(s[group0_indices]), mean(s[group1_indices])
    print("\n    group0_base_rate:", group0_base_rate,"\n    group1_base_rate:", group1_base_rate)
    np.seterr(invalid='ignore')
    print("  COMPAS data loaded")

    return s, y, y_pred_dict, thresholds, group0_base_rate, group1_base_rate, group_indices


def run_ppv_parity_and_for_parity(my_utility_function, thresholds, s, y, y_pred_dict, group_indices):
    optimal_decision_rules = {}

    # Maximize utility

    utilities_a, utilities_b, highest_utility, ideal_threshold_a, ideal_threshold_b = utils.find_highest_utility(
        y=y, s=s, thresholds=thresholds, group_indices=group_indices, utility_function=my_utility_function.calculate_utility)

    metric_values_utility = utils.calculate_metrics(
        utils.metrics_2_prob, y, s, y_pred_dict, ideal_threshold_a, ideal_threshold_b, group_indices)

    print("\nIf we maximize utility without fairness constraints, these are the optimal thresholds:",
          "\n  ideal_threshold_a:", ideal_threshold_a, "\n  ideal_threshold_b:", ideal_threshold_b)
    print("Thereby, we achieve the following fairness values:", metric_values_utility)

    no_fairness_PPV_rate_0, no_fairness_FOR_rate_0 = metric_values_utility["ppv"][0], metric_values_utility["for"][0]
    no_fairness_PPV_rate_1, no_fairness_FOR_rate_1 = metric_values_utility["ppv"][1], metric_values_utility["for"][1]
    acceptance_rate_0, acceptance_rate_1 = metric_values_utility["acceptance_rate"][0], metric_values_utility["acceptance_rate"][1]

    optimal_decision_rules["max_util"] = {
        "ideal_thresholds": (ideal_threshold_a, ideal_threshold_b),
        "no_fairness_PPV non-C": no_fairness_PPV_rate_0,
        "no_fairness_FOR non-C": no_fairness_FOR_rate_0,
        "no_fairness_PPV C": no_fairness_PPV_rate_1,
        "no_fairness_FOR C": no_fairness_FOR_rate_1,
        "acceptance non-C": acceptance_rate_0,
        "acceptance C": acceptance_rate_1
    }

    # Find ideal thresholds under PPV parity

    fairness_function = utils.ppv_prob # similar to the synthetic example, we assume that predicted probabilities are correct, thus, we use the probabilistic PPV rate calculation

    rates_a, rates_b, highest_fair_utility, ideal_rate, ideal_threshold_a, ideal_threshold_b = utils.find_highest_utility_under_fairness(
        y=y, s=s, thresholds=thresholds, group_indices=group_indices, fairness_function=fairness_function, utility_function=my_utility_function.calculate_utility, y_pred_dict=y_pred_dict)

    FOR_rates_a = [utils.forate_prob(y, s, y_pred_dict[t], group_indices[0]) for t in thresholds]
    FOR_rates_b = [utils.forate_prob(y, s, y_pred_dict[t], group_indices[1]) for t in thresholds]

    acceptance_rates_a = [utils.acceptance_rate_prob(y, s, y_pred_dict[t], group_indices[0]) for t in thresholds]
    acceptance_rates_b = [utils.acceptance_rate_prob(y, s, y_pred_dict[t], group_indices[1]) for t in thresholds]

    print("\nUnder PPV parity, these are the optimal thresholds:", "\n  ideal_threshold_a:", ideal_threshold_a, "\n  ideal_threshold_b:", ideal_threshold_b, "\n  PPV rates: ", ideal_rate,
          "\nThis results in the following FOR rates:", "\n  FOR rate group 0:", FOR_rates_a[thresholds.index(ideal_threshold_a)], "\n  FOR rate group 1:", FOR_rates_b[thresholds.index(ideal_threshold_b)])

    PPV_parity_ideal_PPV_rate = ideal_rate
    PPV_parity_ideal_FOR_rate_0 = FOR_rates_a[thresholds.index(ideal_threshold_a)]
    PPV_parity_ideal_FOR_rate_1 = FOR_rates_b[thresholds.index(ideal_threshold_b)]

    acceptance_rate_0 = acceptance_rates_a[thresholds.index(ideal_threshold_a)]
    acceptance_rate_1 = acceptance_rates_b[thresholds.index(ideal_threshold_b)]

    optimal_decision_rules["max_util_PPV_parity"] = {
        "ideal_thresholds": (ideal_threshold_a, ideal_threshold_b),
        "PPV_parity_ideal_PPV_rate": PPV_parity_ideal_PPV_rate,
        "PPV_parity_ideal_FOR non-C": PPV_parity_ideal_FOR_rate_0,
        "PPV_parity_ideal_FOR C": PPV_parity_ideal_FOR_rate_1,
        "acceptance non-C": acceptance_rate_0,
        "acceptance C": acceptance_rate_1
    }

    # Find ideal thresholds for FOR parity

    fairness_function = utils.forate_prob # similar to the synthetic example, we assume that predicted probabilities are correct, thus, we use the probabilistic FOR rate calculation

    rates_a, rates_b, highest_fair_utility, ideal_rate, ideal_threshold_a, ideal_threshold_b = utils.find_highest_utility_under_fairness(
        y=y, s=s, thresholds=thresholds, group_indices=group_indices, fairness_function=fairness_function, utility_function=my_utility_function.calculate_utility, y_pred_dict=y_pred_dict)

    PPV_rates_a = [utils.ppv_prob(y, s, y_pred_dict[t], group_indices[0]) for t in thresholds]
    PPV_rates_b = [utils.ppv_prob(y, s, y_pred_dict[t], group_indices[1]) for t in thresholds]

    acceptance_rates_a = [utils.acceptance_rate_prob(y, s, y_pred_dict[t], group_indices[0]) for t in thresholds]
    acceptance_rates_b = [utils.acceptance_rate_prob(y, s, y_pred_dict[t], group_indices[1]) for t in thresholds]

    print("\nUnder FOR parity, these are the optimal thresholds:", "\n  ideal_threshold_a:", ideal_threshold_a, "\n  ideal_threshold_b:", ideal_threshold_b, "\n  FOR rates: ", ideal_rate,
          ".\nThis results in the following PPV rates:", "\n  PPV rate group 0:", PPV_rates_a[thresholds.index(ideal_threshold_a)], "\n  PPV rate group 1:", PPV_rates_b[thresholds.index(ideal_threshold_b)])

    FOR_parity_ideal_FOR_rate = ideal_rate
    FOR_parity_ideal_PPV_rate_0 = PPV_rates_a[thresholds.index(ideal_threshold_a)]
    FOR_parity_ideal_PPV_rate_1 = PPV_rates_b[thresholds.index(ideal_threshold_b)]

    acceptance_rate_0 = acceptance_rates_a[thresholds.index(ideal_threshold_a)]
    acceptance_rate_1 = acceptance_rates_b[thresholds.index(ideal_threshold_b)]

    optimal_decision_rules["max_util_FOR_parity"] = {
        "ideal_thresholds": (ideal_threshold_a, ideal_threshold_b),
        "FOR_parity_ideal_FOR_rate": FOR_parity_ideal_FOR_rate,
        "FOR_parity_ideal_PPV non-C": FOR_parity_ideal_PPV_rate_0,
        "FOR_parity_ideal_PPV C": FOR_parity_ideal_PPV_rate_1,
        "acceptance non-C": acceptance_rate_0,
        "acceptance C": acceptance_rate_1
    }

    return optimal_decision_rules


def calculate_solutions_for_different_utility_functions(sens_attr):

    threshold_nr = 100

    s, y, y_pred_dict, thresholds, group0_base_rate, group1_base_rate, group_indices = load_compas_data(threshold_nr, sens_attr)

    CASE1_key = (1, -1, -1, 1)  # u_tn, u_fp, u_fn, u_tp
    CASE2_key = (1, -10, -1, 1)  # u_tn, u_fp, u_fn, u_tp
    CASE3_key = (1, -1, -10, 1)  # u_tn, u_fp, u_fn, u_tp

    print("\n   ========== calculate solution for CASE 1: (u_tn, u_fp, u_fn, u_tp) =", str(CASE1_key), "==========")
    CASE1 = {"group0_BR non-C": group0_base_rate, "group1_BR C": group1_base_rate, "optimal_decision_rules":
             run_ppv_parity_and_for_parity(utils.UtilityFunction(*CASE1_key, proba=True), thresholds, s, y, y_pred_dict, group_indices)}

    print("\n   ========== calculate solution for CASE 2: (u_tn, u_fp, u_fn, u_tp) =", str(CASE2_key), "==========")
    CASE2 = {"group0_BR non-C": group0_base_rate, "group1_BR C": group1_base_rate, "optimal_decision_rules":
             run_ppv_parity_and_for_parity(utils.UtilityFunction(*CASE2_key, proba=True), thresholds, s, y, y_pred_dict, group_indices)}

    print("\n   ========== calculate solution for CASE 3: (u_tn, u_fp, u_fn, u_tp) =", str(CASE3_key), "==========")
    CASE3 = {"group0_BR non-C": group0_base_rate, "group1_BR C": group1_base_rate, "optimal_decision_rules":
             run_ppv_parity_and_for_parity(utils.UtilityFunction(*CASE3_key, proba=True), thresholds, s, y, y_pred_dict, group_indices)}

    print("\n\n######### print summary of solution for all 3 cases #########\n")

    for (key, case, case_name) in ((CASE1_key, CASE1, 'CASE1'), (CASE2_key, CASE2, 'CASE2'), (CASE3_key, CASE3, 'CASE3')):
        print("\n[" + case_name + "] TP,FP,FN,TN:", key, "\n")
        for (k, v) in case.items():
            if k != "optimal_decision_rules":
                print("  ", k, ":", round(v, 2))
            else:
                for (k2, v2) in v.items():
                    print("  --", k2, "--")
                    for (k3, v3) in v2.items():
                        print("    ", k3, ":", "\n        non-C:\t", (round(v2["ideal_thresholds"][0][0], 2), round(v2["ideal_thresholds"][0][1], 2)), "\n        C:\t", (round(
                            v2["ideal_thresholds"][1][0], 2), round(v2["ideal_thresholds"][1][1], 2))) if k3 == "ideal_thresholds" else print("    ", k3, ":\t", round(v3, 2))
                        print
        print("\n---\n\n")


def plot_score_distributions(sens_attr):

    threshold_nr = 100

    s, y, y_pred_dict, thresholds, group0_base_rate, group1_base_rate, group_indices = load_compas_data(threshold_nr, sens_attr)

    n = len(s)

    nr_non_caucasian = sum(group_indices[0])
    nr_caucasian = sum(group_indices[1])
    nr_non_caucasian_y1 = sum(y[group_indices[0]])
    nr_caucasian_y1 = sum(y[group_indices[1]])

    print("n =", n)
    print("number of non_caucasian =", nr_non_caucasian)
    print("number of caucasian =", nr_caucasian)
    print("number of non_caucasian with Y=1 =", nr_non_caucasian_y1)
    print("number of caucasian with Y=1 =", nr_caucasian_y1)
    print("base rate non_caucasian =", group0_base_rate)
    print("base rate caucasian =", group1_base_rate)

    hist_data = pd.concat([s, group_indices[1]], axis=1)
    hist_data = hist_data.rename(columns={0: "Score", "race": "Race"})
    hist_data['Race'] = hist_data['Race'].replace([True, False], ['Caucasian', 'Non-caucasian'])
    sns.set_context('paper', font_scale=1.3)
    sns.histplot(data=hist_data, x="Score", hue="Race",element="step", stat="density", fill=False)

    # plot thresholds to visualize unconstrained optimum
    plt.axvline(0.5, color="red", linestyle="--")  # case 1
    plt.axvline(0.85, color="red", linestyle="--")  # case 2
    plt.axvline(0.15, color="red", linestyle="--")  # case 3

    plt.xticks((0.0, 0.15, 0.2, 0.4, 0.5, 0.6, 0.8, 0.85, 1.0), ("0.0", r"$t_{u3}$", "0.2", "0.4", r"$t_{u1}$", "0.6", "0.8", r"$t_{u2}$", "1.0"))
    # plt.show()
    plt.savefig("figures/score_distributions.pdf")
    plt.clf()


sens_attr = "race"
calculate_solutions_for_different_utility_functions(sens_attr)
# plot_score_distributions(sens_attr)
