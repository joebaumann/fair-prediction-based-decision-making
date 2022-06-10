import numpy as np
import pandas as pd
import utils
from scipy.stats import beta
from statistics import mean


def get_parameters(population):
    if population == "population1":
        # POPULATION 1 -> two lower-bound thresholds
        # group 0 beta distribution parameters
        group0_size, group0_a, group0_b = 20000, 1.9, 1.35
        # group 1 beta distribution parameters
        group1_size, group1_a, group1_b = 20000, 3, 2
        fig_name = "figures/population1.pdf"

    if population == "population2":
        # POPULATION 2 -> disadvantaged group is held to a higher standard
        # group 0 beta distribution parameters
        group0_size, group0_a, group0_b = 20000, 2, 3
        # group 1 beta distribution parameters
        group1_size, group1_a, group1_b = 20000, 3, 2
        fig_name = "figures/population2.pdf"

    if population == "population3":
        # POPULATION 3 -> one upper- and one lower-bound threshold
        # group 0 beta distribution parameters
        group0_size, group0_a, group0_b = 20000, 2, 3
        # group 1 beta distribution parameters
        group1_size, group1_a, group1_b = 2000, 3, 2
        fig_name = "figures/population3.pdf"

    # synthetic data example: positive utility for Y=1 and negative utility for Y=0
    u_tn, u_fp, u_fn, u_tp = 0, -3, 0, 7

    # generate the utility function based on the specified utility values
    my_utility_function = utils.UtilityFunction(u_tn, u_fp, u_fn, u_tp, proba=True)

    # derive the unconstrained optimal decision rule
    optimal_unconstrained_decision_rule = my_utility_function.get_optimal_unconstrained_decision_rule()

    return group0_size, group0_a, group0_b, group1_size, group1_a, group1_b, fig_name, my_utility_function, optimal_unconstrained_decision_rule


def generate_synthetic_data(group0_size, group0_a, group0_b, group1_size, group1_a, group1_b, threshold_nr, optimal_unconstrained_decision_rule):

    all_indices = pd.Series(range(1, group0_size + group1_size + 1))
    group0_indices = all_indices <= group0_size
    group1_indices = all_indices > group0_size

    group_indices = [group0_indices, group1_indices]

    # generate probability distributions for both groups convert to pd Series with previously defined indices
    s_group0 = pd.Series(beta.rvs(group0_a, group0_b, loc=0, scale=1, size=group0_size,
                                  random_state=utils.seed), index=group0_indices[group0_indices].index)
    s_group1 = pd.Series(beta.rvs(group1_a, group1_b, loc=0, scale=1, size=group1_size,
                                  random_state=utils.seed), index=group1_indices[group1_indices].index)

    s = s_group0.append(s_group1, verify_integrity=True)
    y = pd.Series(np.random.binomial(1, s), index=s.index)
    thresholds = utils.generate_thresholds(threshold_nr)
    y_pred_dict = {t: s.between(t[0], t[1]) for t in thresholds}

    print("Synthetic data generated.")
    group0_base_rate, group1_base_rate = mean(s[group0_indices]), mean(s[group1_indices])
    print("  base rate group 0:", group0_base_rate, "\n  base rate group 1:", group1_base_rate)

    np.seterr(invalid='ignore')

    return s, y, y_pred_dict, thresholds, group0_base_rate, group1_base_rate, group_indices
