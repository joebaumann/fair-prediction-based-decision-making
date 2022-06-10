import sys
import matplotlib.pyplot as plt
import numpy as np
import data.load_synthetic_data as load_synthetic_data
import utils

if len(sys.argv) < 2:
    print('Argument missing. Please provide one of those three population names: population1, population2, population3')
else:
    POPULATION = sys.argv[1]
    print('Running the script for:', POPULATION)

# get the parameters for the sepcified population
group0_size, group0_a, group0_b, group1_size, group1_a, group1_b, fig_name, my_utility_function, optimal_unconstrained_decision_rule = load_synthetic_data.get_parameters(
    POPULATION)

threshold_nr = 100

# load the synthetic dataset with the population parameters
s, y, y_pred_dict, thresholds, group0_base_rate, group1_base_rate, group_indices = load_synthetic_data.generate_synthetic_data(
    group0_size, group0_a, group0_b, group1_size, group1_a, group1_b, threshold_nr, optimal_unconstrained_decision_rule)


# Maximize utility without any fairness constraint

utilities_a, utilities_b, highest_utility, ideal_threshold_a, ideal_threshold_b = utils.find_highest_utility(
    y=y, s=s, thresholds=thresholds, group_indices=group_indices, utility_function=my_utility_function.calculate_utility)

metric_values_utility = utils.calculate_metrics(
    utils.metrics_2_prob, y, s, y_pred_dict, ideal_threshold_a, ideal_threshold_b, group_indices)

print("\nIf we maximize utility without fairness constraints, these are the optimal thresholds:",
      "\n  ideal_threshold_a:", ideal_threshold_a, "\n  ideal_threshold_b:", ideal_threshold_b)
print("Thereby, we achieve the following fairness values:",
      metric_values_utility, "\nThis yields a total utility of:", highest_utility)

no_fairness_PPV_rate_0, no_fairness_FOR_rate_0 = metric_values_utility[
    "ppv"][0], metric_values_utility["for"][0]
no_fairness_PPV_rate_1, no_fairness_FOR_rate_1 = metric_values_utility[
    "ppv"][1], metric_values_utility["for"][1]


# Find optimal decision rule for PPV parity

fairness_function = utils.ppv_prob

rates_a, rates_b, highest_fair_utility, ideal_rate, ideal_threshold_a, ideal_threshold_b = utils.find_highest_utility_under_fairness(
    y=y, s=s, thresholds=thresholds, group_indices=group_indices, fairness_function=fairness_function, utility_function=my_utility_function.calculate_utility, y_pred_dict=y_pred_dict)

FOR_rates_a = [utils.forate_prob(y, s, y_pred_dict[t], group_indices[0]) for t in thresholds]
FOR_rates_b = [utils.forate_prob(y, s, y_pred_dict[t], group_indices[1]) for t in thresholds]

print("\nUnder PPV parity, these are the optimal thresholds:", "\n  ideal_threshold_a:", ideal_threshold_a, "\n  ideal_threshold_b:", ideal_threshold_b, "\nResulting PPV rates:", ideal_rate,
      ". This results in the following FOR rates:\n  FOR rate group 0:", FOR_rates_a[thresholds.index(ideal_threshold_a)], "\n  FOR rate group 1:", FOR_rates_b[thresholds.index(ideal_threshold_b)], "\nThis yields a total utility of:", highest_fair_utility)

PPV_parity_ideal_PPV_rate = ideal_rate
PPV_parity_ideal_FOR_rate_0 = FOR_rates_a[thresholds.index(ideal_threshold_a)]
PPV_parity_ideal_FOR_rate_1 = FOR_rates_b[thresholds.index(ideal_threshold_b)]

# Find optimal decision rule for FOR parity

fairness_function = utils.forate_prob

rates_a, rates_b, highest_fair_utility, ideal_rate, ideal_threshold_a, ideal_threshold_b = utils.find_highest_utility_under_fairness(
    y=y, s=s, thresholds=thresholds, group_indices=group_indices, fairness_function=fairness_function, utility_function=my_utility_function.calculate_utility, y_pred_dict=y_pred_dict)

PPV_rates_a = [utils.ppv_prob(
    y, s, y_pred_dict[t], group_indices[0]) for t in thresholds]
PPV_rates_b = [utils.ppv_prob(
    y, s, y_pred_dict[t], group_indices[1]) for t in thresholds]

print("\nUnder FOR parity, these are the optimal thresholds:", "\n  ideal_threshold_a:", ideal_threshold_a, "\n  ideal_threshold_b:", ideal_threshold_b, "\nResulting FOR rates:", ideal_rate,
      "\nThis results in the following PPV rates:\n  PPV rate group 0:", PPV_rates_a[thresholds.index(ideal_threshold_a)], "\n  PPV rate group 1:", PPV_rates_b[thresholds.index(ideal_threshold_b)], "\nThis yields a total utility of:", highest_fair_utility)

FOR_parity_ideal_FOR_rate = ideal_rate
FOR_parity_ideal_PPV_rate_0 = PPV_rates_a[thresholds.index(ideal_threshold_a)]
FOR_parity_ideal_PPV_rate_1 = PPV_rates_b[thresholds.index(ideal_threshold_b)]


# Find optimal decision rule under sufficiency

fairness_functions = (utils.ppv_prob, utils.forate_prob)

rates_a_PPV, rates_a_FOR, rates_b_PPV, rates_b_FOR = utils.get_PPV_FOR_rates_for_all_decision_rules(
    y=y, s=s, thresholds=thresholds, group_indices=group_indices, fairness_functions=fairness_functions, utility_function=my_utility_function.calculate_utility, y_pred_dict=y_pred_dict)

optimal_solution, max_base_rate, min_base_rate, solution_space_without_None_values, rates_a_PPV_without_None_values, turning_point_t_index, min_FOR, max_FOR = utils.generate_solution_space_and_maximize_utility(
    y, group0_base_rate, group1_base_rate, rates_a_PPV, rates_a_FOR, rates_b_PPV, rates_b_FOR, thresholds, group_indices, my_utility_function, optimal_unconstrained_decision_rule, s, y_pred_dict)

optimal_solution_total_utility = [v["total_utility"] for (k, v) in optimal_solution.items()]
optimal_solution_PPV_rates = [v["PPV_rate"] for (k, v) in optimal_solution.items()]
optimal_solution_FOR_rates = [v["FOR_rate"] for (k, v) in optimal_solution.items()]
optimal_solution_threshold_a = [v["threshold_a"] for (k, v) in optimal_solution.items()]
optimal_solution_threshold_b = [v["threshold_b"] for (k, v) in optimal_solution.items()]
optimal_solution_randomization_a = [v["randomization_a"] for (k, v) in optimal_solution.items()]
optimal_solution_randomization_b = [v["randomization_b"] for (k, v) in optimal_solution.items()]

index_of_utils = np.argmax(optimal_solution_total_utility)

print("\nUnder sufficiency, these are the optimal thresholds:", "\n  ideal_threshold_a:", optimal_solution_threshold_a[index_of_utils], "randomization_a:", optimal_solution_randomization_a[index_of_utils], "\n  ideal_threshold_b:", optimal_solution_threshold_b[index_of_utils], "randomization_b:",
      optimal_solution_randomization_b[index_of_utils], "\nThis results in the following rates (for both groups!):\n  PPV:", optimal_solution_PPV_rates[index_of_utils], "\n  FOR:", optimal_solution_FOR_rates[index_of_utils], "\nThis yields a total utility of:", optimal_solution_total_utility[index_of_utils])
print("\nExplanation of randomized decision rules: A threshold of (t1, t2) with randomization q means: Individuals with p<t1 or p>t2 are not selected (i.e., assigned decision D=0). Individuals with t1<p<t2 are selected with probability q.")


# plot the solution

if POPULATION == "population3":
    figsize = (6.7, 4)
else:
    figsize = (5, 4)

fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(1, 1, 1)

ax.set_facecolor((128/255, 128/255, 128/255, 0.4))

ax.plot(rates_a_PPV, rates_a_FOR, linestyle="-", color="orange", label="$F_0(PPV_{A=0})$", alpha=0.6)
ax.plot(rates_b_PPV, rates_b_FOR, linestyle="-", color="blue", label="$F_1(PPV_{A=1})$", alpha=0.6)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("PPV")
ax.set_ylabel("FOR")
ax.axhline(group0_base_rate, color="orange", alpha=0.3, linestyle="--")
ax.axvline(group0_base_rate, color="orange", alpha=0.3, linestyle="--")

ax.axhline(group1_base_rate, color="blue", alpha=0.3, linestyle="--")
ax.axvline(group1_base_rate, color="blue", alpha=0.3, linestyle="--")

# add patch for solution space
plt.fill_between(rates_a_PPV_without_None_values[:turning_point_t_index], min_FOR[:turning_point_t_index], max_base_rate, where=(
    (rates_a_PPV_without_None_values[:turning_point_t_index] < min_base_rate) & (min_FOR[:turning_point_t_index] > max_base_rate)), interpolate=True, color='white')
plt.fill_between(rates_a_PPV_without_None_values[turning_point_t_index:], max_FOR[turning_point_t_index:], min_base_rate, where=(
    (rates_a_PPV_without_None_values[turning_point_t_index:] > max_base_rate) & (max_FOR[turning_point_t_index:] < min_base_rate)), interpolate=True, color='white')

# plot optimal solution without fairness for both groups
ax.plot(no_fairness_PPV_rate_0, no_fairness_FOR_rate_0, markeredgecolor='#ff5500', markerfacecolor='orange',
        marker="P", markersize=10, label="no fairness group 0", alpha=0.7, linestyle='None')
ax.plot(no_fairness_PPV_rate_1, no_fairness_FOR_rate_1, markeredgecolor='#00ffff', markerfacecolor='blue',
        marker="P", markersize=10, label="no fairness group 1", alpha=0.7, linestyle='None')

# PPV parity
ax.plot(PPV_parity_ideal_PPV_rate, PPV_parity_ideal_FOR_rate_0, markeredgecolor='#ff5500',
        markerfacecolor='orange', marker='d', markersize=10, label="PPV parity group 0", alpha=0.7, linestyle='None')
ax.plot(PPV_parity_ideal_PPV_rate, PPV_parity_ideal_FOR_rate_1, markeredgecolor='#00ffff',
        markerfacecolor='blue', marker='d', markersize=10, label="PPV parity group 1", alpha=0.7, linestyle='None')

# FOR parity
ax.plot(FOR_parity_ideal_PPV_rate_0, FOR_parity_ideal_FOR_rate, markeredgecolor='#ff5500',
        markerfacecolor='orange', marker='h', markersize=10, label="FOR parity group 0", alpha=0.7, linestyle='None')
ax.plot(FOR_parity_ideal_PPV_rate_1, FOR_parity_ideal_FOR_rate, markeredgecolor='#00ffff',
        markerfacecolor='blue', marker='h', markersize=10, label="FOR parity group 1", alpha=0.7, linestyle='None')

# plot sufficiency solution
ax.plot(optimal_solution_PPV_rates[index_of_utils], optimal_solution_FOR_rates[index_of_utils],
        markerfacecolor='red', marker='*', markersize=8, label="sufficiency", alpha=0.9, linestyle='None')

if POPULATION == "population3":
    ax.legend(bbox_to_anchor=(1.05, 0.9), labelspacing=1)

plt.style.use('seaborn-paper')
plt.tight_layout()

plt.show(block=False)

# plt.show()
fig.savefig(fig_name, bbox_inches='tight')
