import pickle
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import utils


### CONFIG ###
sens_attr = "race"
threshold_nr = 100
fairness_function, fairness_metric_name = utils.fpr, "fpr"
# partial fulfillment of the fairness metric
disparity_function = "diff" # alternative options: "p_percentage" or "p_percentage_LEQ"
# specify the different fairness constraints to be evaluated
fairness_constraints = np.linspace(1/6, 1-(1/6), 6) # this can be any list of numbers between 0 and 1
# define the utility_function_weights: u_tn, u_fp, u_fn, u_tp
utility_function_weights = (1, 0, 0, 1)  # accuracy
EVAL = "eval_on_test_set"  # alternative option: "eval_on_train_set"
# This is needed to make sure that this threshold is also included in the generated post-processing thresholds
probable_unconstrained_optimal_threshold = utils.UtilityFunction(*utility_function_weights, proba=False, normalize=True).get_optimal_unconstrained_decision_rule()
filename = "output_utility_fairness_tradeoff/performance_fairness_tradeff_10_folds_" + disparity_function + ".txt"
figure_name, figure_name_2 = "utility_fairness_tradeoff_" + disparity_function + ".pdf", "utility_fairness_tradeoff_FPRs_" + disparity_function + ".pdf"


def load_compas_data(threshold_nr, sens_attr, probable_unconstrained_optimal_threshold, fairness_metric_name="fpr", plot_ROC=False):
    print("\nStarting to load COMPAS data")
    data_all_folds = {}
    filename = "data/propublica-recidivism_10_FOLDS_preprocessed.pickle"
    with open(filename, 'rb') as f:
        all_folds_unpickled = pickle.load(f)

    for fold in all_folds_unpickled.keys():
        my_dict_unpickled = all_folds_unpickled[fold]
        my_dict_unpickled["privileged_val"] = "Caucasian"
        group0_indices_train = my_dict_unpickled["train_df_sensitive_attrs"][sens_attr] != my_dict_unpickled["privileged_val"]
        group1_indices_train = my_dict_unpickled["train_df_sensitive_attrs"][sens_attr] == my_dict_unpickled["privileged_val"]
        group0_indices_test = my_dict_unpickled["test_df_sensitive_attrs"][sens_attr] != my_dict_unpickled["privileged_val"]
        group1_indices_test = my_dict_unpickled["test_df_sensitive_attrs"][sens_attr] == my_dict_unpickled["privileged_val"]

        #print("  Groups have been defined: group_0 is NOT", str(my_dict_unpickled["privileged_val"]), "and group_1 is", str(my_dict_unpickled["privileged_val"]))

        group_indices_train = [group0_indices_train, group1_indices_train]
        group_indices_test = [group0_indices_test, group1_indices_test]

        s_train = my_dict_unpickled["predictions_proba_X_train"]
        y_train = my_dict_unpickled["y_train"]
        s_test = my_dict_unpickled["predictions_proba_X_test"]
        y_test = my_dict_unpickled["y_test"]

        if fairness_metric_name in ['ppv', 'for']:
            # optimal decision rules under PPV parity or FOR parity always take the form of group-specific upper- or lower-bound thresholds
            thresholds = utils.generate_upper_and_lower_bound_thresholds(s_train, threshold_nr, probable_unconstrained_optimal_threshold)
            y_pred_dict = {t: s_train.between(t[0], t[1]) for t in thresholds}
            y_pred_dict_test = {t: s_test.between(t[0], t[1]) for t in thresholds}

        else:
            # optimal decision rules under (conditional) statistical parity, TPR parity, FPR parity, always take the form of group-specific lower-bound thresholds
            thresholds = utils.generate_lower_bound_thresholds(
                s_train, threshold_nr, probable_unconstrained_optimal_threshold)
            y_pred_dict = {t: s_train >= t for t in thresholds}
            y_pred_dict_test = {t: s_test >= t for t in thresholds}

        group0_base_rate, group1_base_rate = mean(
            s_train[group0_indices_train]), mean(s_train[group1_indices_train])

        np.seterr(invalid='ignore')

        def ROC_curves():
            print('fold', fold, 'roc_auc_score for LR:\n  group 0 = Afr.Amer.:\t', roc_auc_score(
                y_test[group0_indices_test], s_test[group0_indices_test]), '\n  group 1 = Cauca.:\t', roc_auc_score(
                y_test[group1_indices_test], s_test[group1_indices_test]))
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(
                y_test[group0_indices_test], s_test[group0_indices_test])
            false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(
                y_test[group1_indices_test], s_test[group1_indices_test])
            plt.subplots(1, figsize=(10, 10))
            plt.title('Receiver Operating Characteristic [fold: ' + str(fold) + ']')
            plt.plot(false_positive_rate1, true_positive_rate1, label="group 0 [NON-CAUCASIAN]")
            plt.plot(false_positive_rate2, true_positive_rate2, label="group 1 [CAUCASIAN]")
            plt.plot([0, 1], ls="--", label="random")
            plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.legend()
            plt.show()

        if plot_ROC == True:
            ROC_curves()

        data_all_folds[fold] = [s_train, s_test, y_train, y_test, y_pred_dict, y_pred_dict_test,
                                thresholds, group0_base_rate, group1_base_rate, group_indices_train, group_indices_test]

    print("COMPAS data loaded.")

    return data_all_folds


def plot_ROC_curves(threshold_nr, sens_attr, probable_unconstrained_optimal_threshold):
    load_compas_data(threshold_nr, sens_attr, probable_unconstrained_optimal_threshold, plot_ROC=True)


def derive_optimal_decision_rules(my_utility_function, fairness_function, fairness_metric_name, s_train, s_test, y_train, y_test, y_pred_dict, y_pred_dict_test, thresholds, group_indices_train, group_indices_test, fairness_constraints, disparity_function):

    optimal_decision_rules = {}

    utility_function = my_utility_function.calculate_utility

    # Maximize utility

    utilities_a, utilities_b, highest_utility, ideal_threshold_a, ideal_threshold_b = utils.find_highest_utility(
        y=y_train, s=s_train, thresholds=thresholds, group_indices=group_indices_train, utility_function=utility_function, y_pred_dict=y_pred_dict)

    metric_values_utility = utils.calculate_metrics(
        utils.metrics_3, y_train, s_train, y_pred_dict, ideal_threshold_a, ideal_threshold_b, group_indices_train)

    no_FC_fairness_rate_0, no_FC_fairness_rate_1 = metric_values_utility[
        fairness_metric_name][0], metric_values_utility[fairness_metric_name][1]

    #print("\nIf we maximize utility without fairness constraints, these are the optimal thresholds:", "ideal_threshold_a [NON-CAUCASIAN]:", ideal_threshold_a, " / ideal_threshold_b [CAUCASIAN]:", ideal_threshold_b)
    #print("Thereby, we achieve the following utility:", highest_utility)
    #print("Thereby, we achieve the following fairness values (", fairness_metric_name, "): group 0 [NON-CAUCASIAN] =", no_FC_fairness_rate_0, " /// group 1 [CAUCASIAN] =", no_FC_fairness_rate_1)
    #print("All metrics:", metric_values_utility)

    # max_util_without_fairness_constraint
    optimal_decision_rules[0.0] = {
        "fairness_constraint": 0.0,
        "utility": highest_utility,
        "thresholds": (ideal_threshold_a, ideal_threshold_b),
        "fairness_metric_name": fairness_metric_name,
        "fairness_rate_0": no_FC_fairness_rate_0,
        "fairness_rate_1": no_FC_fairness_rate_1,
        "all_metrics": metric_values_utility
    }

    # Find ideal thresholds while satisfying fairness

    rates_a, rates_b, highest_fair_utility, ideal_rate, ideal_threshold_a, ideal_threshold_b = utils.find_highest_utility_under_fairness(
        y=y_train, s=s_train, thresholds=thresholds, group_indices=group_indices_train, fairness_function=fairness_function, utility_function=utility_function, y_pred_dict=y_pred_dict)

    metric_values_fair = utils.calculate_metrics(
        utils.metrics_3, y_train, s_train, y_pred_dict, ideal_threshold_a, ideal_threshold_b, group_indices_train)

    full_FC_fairness_rate_0, full_FC_fairness_rate_1 = metric_values_fair[
        fairness_metric_name][0], metric_values_fair[fairness_metric_name][1]

    #print("\nUnder", fairness_metric_name, "parity, these are the optimal thresholds:", "ideal_threshold_a [NON-CAUCASIAN]:", ideal_threshold_a, " / ideal_threshold_b [CAUCASIAN]:", ideal_threshold_b)
    #print("Thereby, we achieve the following utility:", highest_fair_utility)
    #print("Thereby, we achieve the following fairness values: group 0 [NON-CAUCASIAN] =", full_FC_fairness_rate_0, " /// group 1 [CAUCASIAN] =", full_FC_fairness_rate_1)
    #print("All metrics:", metric_values_fair)

    # max_util_under_fairness
    optimal_decision_rules[1.0] = {
        "fairness_constraint": 1.0,
        "utility": highest_fair_utility,
        "thresholds": (ideal_threshold_a, ideal_threshold_b),
        "fairness_metric_name": fairness_metric_name,
        "fairness_rate_0": full_FC_fairness_rate_0,
        "fairness_rate_1": full_FC_fairness_rate_1,
        "all_metrics": metric_values_fair
    }

    counter = 0

    for fairness_constraint in fairness_constraints:
        counter += 1
        print(fairness_metric_name, "-", counter,
              "- fairness_constraint: ", fairness_constraint)
        highest_fairness_constraint_utility, ideal_rate, ideal_fairness_constraint_threshold_a, ideal_fairness_constraint_threshold_b = utils.find_highest_utility_under_degree_of_fairness_constraint(
            fairness_constraint, metric_values_utility, metric_values_fair, rates_a, rates_b, y_train, s_train, thresholds, group_indices_train, fairness_metric_name, utility_function=utility_function, y_pred_dict=y_pred_dict, disparity_function=disparity_function)
        metric_values_fairness_constraint = utils.calculate_metrics(
            utils.metrics_3, y_train, s_train, y_pred_dict, ideal_fairness_constraint_threshold_a, ideal_fairness_constraint_threshold_b, group_indices_train)

        fairness_rate_0, fairness_rate_1 = metric_values_fairness_constraint[
            fairness_metric_name][0], metric_values_fairness_constraint[fairness_metric_name][1]

        #print("\nUnder", fairness_metric_name, "parity, these are the optimal thresholds:", "ideal_threshold_a [NON-CAUCASIAN]:", ideal_fairness_constraint_threshold_a, " / ideal_threshold_b [CAUCASIAN]:", ideal_fairness_constraint_threshold_b)
        #print("Thereby, we achieve the following utility:", highest_fairness_constraint_utility)
        #print("Thereby, we achieve the following fairness values: group 0 [NON-CAUCASIAN] =", fairness_rate_0, " /// group 1 [CAUCASIAN] =", fairness_rate_1)
        #print("All metrics:", metric_values_fairness_constraint)

        optimal_decision_rules[round(fairness_constraint, 3)] = {
            "fairness_constraint": round(fairness_constraint, 3),
            "utility": highest_fairness_constraint_utility,
            "thresholds": (ideal_fairness_constraint_threshold_a, ideal_fairness_constraint_threshold_b),
            "fairness_metric_name": fairness_metric_name,
            "fairness_rate_0": fairness_rate_0,
            "fairness_rate_1": fairness_rate_1,
            "all_metrics": metric_values_fairness_constraint
        }

    # now go through the post processed solutions, apply the thresholds to the TEST SET and evaluate the performance and the fairness

    optimal_decision_rules_evalOnTest = {}
    optimal_decision_rules_evalOnTest = {}

    print("\n  -- evaluate decision rules on test set --")

    for fairness_constraint, result in optimal_decision_rules.items():
        print("  fairness_constraint -", fairness_constraint)
        optimal_thresholds = result["thresholds"]
        metric_values_test = utils.calculate_metrics(
            utils.metrics_3, y_test, s_train, y_pred_dict_test, optimal_thresholds[0], optimal_thresholds[1], group_indices_test)
        fairness_rate_0, fairness_rate_1 = metric_values_test[
            fairness_metric_name][0], metric_values_test[fairness_metric_name][1]
        y_pred = utils.y_pred_two_thresholds(
            s_test, *group_indices_test, optimal_thresholds[0], optimal_thresholds[1])
        utility = utility_function(y_test, None, y_pred)

        optimal_decision_rules_evalOnTest[round(fairness_constraint, 3)] = {
            "fairness_constraint": round(fairness_constraint, 3),
            "utility": utility,
            "thresholds": optimal_thresholds,
            "fairness_metric_name": fairness_metric_name,
            "fairness_rate_0": fairness_rate_0,
            "fairness_rate_1": fairness_rate_1,
            "all_metrics": metric_values_test
        }

    return {"eval_on_test_set": optimal_decision_rules_evalOnTest, "eval_on_train_set": optimal_decision_rules}


def calculate_fairness_accuracy_tradeoff(filename, threshold_nr, sens_attr, utility_function_weights, fairness_function, fairness_metric_name, fairness_constraints, disparity_function, probable_unconstrained_optimal_threshold):

    all_solutions = {}

    my_utility_function = utils.UtilityFunction(*utility_function_weights, proba=False, normalize=True)
    p_zero_utility = my_utility_function.get_optimal_unconstrained_decision_rule()

    data_all_folds = load_compas_data(
        threshold_nr, sens_attr, probable_unconstrained_optimal_threshold, fairness_metric_name)

    for fold, fold_data in data_all_folds.items():
        print("\n=========== FOLD nr:", fold, "===========\n")
        print("  -- derive optimal decision rules based on train set --")
        s_train, s_test, y_train, y_test, y_pred_dict, y_pred_dict_test, thresholds, group0_base_rate, group1_base_rate, group_indices_train, group_indices_test = fold_data
        all_solutions[fold] = derive_optimal_decision_rules(my_utility_function, fairness_function, fairness_metric_name, s_train, s_test, y_train,
                                                            y_test, y_pred_dict, y_pred_dict_test, thresholds, group_indices_train, group_indices_test, fairness_constraints, disparity_function)

    with open(filename, 'wb') as fp:
        pickle.dump(all_solutions, fp)

    return all_solutions


def plot_pareto_frontier(filename, EVAL, disparity_function):

    print("\nPlotting the results...\n")

    plt.figure(figsize=(3.5, 3.5))

    with open(filename, 'rb') as f:
        all_folds_results = pickle.load(f)

    all_utilities = []
    all_fairness_scores = []
    mean_utilities = []
    mean_fairness_scores = []
    fairness_constraints = sorted(all_folds_results[0][EVAL].keys())
    for constraint in fairness_constraints:
        utilities = [fold_data[EVAL][constraint]["utility"]
                     for fold_data in all_folds_results.values()]
        fairness_scores = [1-abs(fold_data[EVAL][constraint]["fairness_rate_0"]-fold_data[EVAL]
                                 [constraint]["fairness_rate_1"]) for fold_data in all_folds_results.values()]
        print("--- gamma:", constraint, "---")
        print("  MEAN DM UTILITY:", mean(utilities))
        print("  FAIRNESS MEANS: FPR_Afr.Amer.=", mean([fold_data[EVAL][constraint]["fairness_rate_0"] for fold_data in all_folds_results.values()]), "/ FPR_Cauca.=", mean(
            [fold_data[EVAL][constraint]["fairness_rate_1"] for fold_data in all_folds_results.values()]), "/ mean absolute fairness difference:", mean(fairness_scores))
        print("  DECISION RULES (average thresholds): Thresh.Afr.Amer.=", mean([fold_data[EVAL][constraint]["thresholds"][0] for fold_data in all_folds_results.values(
        )]), "/ Thresh.Cauca.=", mean([fold_data[EVAL][constraint]["thresholds"][1] for fold_data in all_folds_results.values()]))
        mean_utilities.append(mean(utilities))
        mean_fairness_scores.append(mean(fairness_scores))
        if constraint == 0.0:
            plt.plot(mean(fairness_scores), mean(utilities),
                     "Xr", ms=12, label=r'$d_{unconstrained}$')
        if constraint == 1.0:
            plt.plot(mean(fairness_scores), mean(utilities),
                     "og", ms=12, label=r'$d_{fair}$')
        all_utilities.extend(utilities)
        all_fairness_scores.extend(fairness_scores)

    plt.plot(mean_fairness_scores, mean_utilities, lw=3)
    plt.ylabel('Performance: $U_{DM}$')
    plt.xlabel('Fairness: $FPR_{G=a}-FPR_{G=c}$')
    plt.legend(fontsize=12)
    plt.xlim((0.855, 0.982))
    plt.ylim((0.662, 0.6685))
    plt.xticks([0.875, 0.9, 0.925, 0.95])
    plt.style.use('seaborn-paper')
    plt.savefig("figures/" + figure_name, bbox_inches='tight')
    plt.show()


def plot_gamma_and_fairness(filename, EVAL, disparity_function):

    plt.figure(figsize=(3.5, 3.5))

    with open(filename, 'rb') as f:
        all_folds_results = pickle.load(f)

    all_utilities = []
    all_fairness_scores = []
    mean_utilities = []
    mean_fairness_scores = []
    mean_metric_scores_AFRICAN_AMERICANS = []
    mean_metric_scores_CAUCASIANS = []
    fairness_constraints = sorted(all_folds_results[0][EVAL].keys())
    for constraint in fairness_constraints:
        utilities = [fold_data[EVAL][constraint]["utility"]
                     for fold_data in all_folds_results.values()]
        fairness_scores = [1-abs(fold_data[EVAL][constraint]["fairness_rate_0"]-fold_data[EVAL]
                                 [constraint]["fairness_rate_1"]) for fold_data in all_folds_results.values()]

        metric_scores_AFRICAN_AMERICANS = [
            fold_data[EVAL][constraint]["fairness_rate_0"] for fold_data in all_folds_results.values()]
        metric_scores_CAUCASIANS = [
            fold_data[EVAL][constraint]["fairness_rate_1"] for fold_data in all_folds_results.values()]

        mean_metric_scores_AFRICAN_AMERICANS.append(
            mean(metric_scores_AFRICAN_AMERICANS))
        mean_metric_scores_CAUCASIANS.append(mean(metric_scores_CAUCASIANS))

        print("constraint:", constraint)
        print("mean utility:", mean(utilities))
        print("mean absolute fairness difference:", mean(fairness_scores), "\n")
        mean_utilities.append(mean(utilities))
        mean_fairness_scores.append(mean(fairness_scores))

        all_utilities.extend(utilities)
        all_fairness_scores.extend(fairness_scores)

    plt.plot(fairness_constraints, mean_fairness_scores, lw=3,
             label=r'$min\left(\frac{FPR_{G=a}}{FPR_{G=c}}, \frac{FPR_{G=c}}{FPR_{G=a}}\right)$')
    plt.plot(fairness_constraints, mean_metric_scores_AFRICAN_AMERICANS,
             lw=3, label='$FPR_{G=a}$')  # FPR [AFRICAN-AMERICAN - group 0]
    plt.plot(fairness_constraints, mean_metric_scores_CAUCASIANS,
             lw=3, label=r'$FPR_{G=c}$')  # FPR [CAUCASIAN - group 1]
    plt.xlabel('$\gamma$')
    plt.ylabel('FPR')
    plt.legend(fontsize=14, loc=(0.07, 0.43))
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.style.use('seaborn-paper')
    plt.savefig("figures/" + figure_name_2, bbox_inches='tight')
    plt.show()

#plot_ROC_curves(threshold_nr, sens_attr, probable_unconstrained_optimal_threshold)

fairness_and_performance_of_all_folds = calculate_fairness_accuracy_tradeoff(filename, threshold_nr, sens_attr, utility_function_weights, fairness_function, fairness_metric_name, fairness_constraints, disparity_function, probable_unconstrained_optimal_threshold)

plot_pareto_frontier(filename, EVAL, disparity_function)

plot_gamma_and_fairness(filename, EVAL, disparity_function)
