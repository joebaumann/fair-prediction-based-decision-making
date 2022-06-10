# Fair prediction-based decision making

This repository provides an implementation for prediction based decision making under group fairness in python for the post-processing techniques introduced in our [FAccT'22](https://arxiv.org/abs/2206.02237) and SDS'2022 papers.

## Conda virtual environment setup

Install the necessary requirements with conda:
```
conda env create -f environment.yml
conda activate fair-prediction-based-decision-making
```

Python version: ``3.8.10``

## Using the code

This repository contains code to evaluate and implement the fairness of prediction-based decision making.
Specifically, it is applied to two different datasets:

### Synthetic data example

The script [example_synthetic.py](example_synthetic.py) contains the code to derive optimal decision rules that satisfy sufficiency, PPV parity, or FOR parity for three different synthetic populations.
In addition, the plots that visualize these different solutions are generated in the end.
The plots are saved in the [figures](figures) folder.

Run the scripts with:
```
python example_synthetic.py population1
python example_synthetic.py population2
python example_synthetic.py population3
```

### COMPAS data example

In addition to the synthetic dataset, this repository contains an analysis of the COMPAS dataset.
The COMPAS dataset can be found in the [data](data) folder and is taken from [Friedler et al. (2018)](https://github.com/algofairness/fairness-comparison/tree/master/fairness/data/preprocessed).

To preprocess the COMPAS dataset, run:
```
python data/load_compas_data.py <NUM_OF_FOLDS> <DROP_NON_CAUCASIAN_NON_AFRICAN_AMERICAN> <FILENAME_ADDITION>
```
where the command line arguments denote the following:
- *<NUM_OF_FOLDS>*: denotes the number of folds.
- *<DROP_NON_CAUCASIAN_NON_AFRICAN_AMERICAN>* is a boolean that specifies whether all individuals or only African American and Caucasian individuals should be considered.
- *<FILENAME_ADDITION>*: adds a string to the pickled file containing the preprocessed dataset.

#### Enforcing PPV parity

As we show in our [FAccT'22](https://arxiv.org/abs/2206.02237) paper, optimal fair decision rules depend on the decision maker's utility function.
To evaluate the effect of enforcing PPV parity (aka predictive parity) or FOR parity as a hard fairness constraint for the COMPAS dataset (comparing Caucasian individuals with non-Caucasian individuals), first preprocess the data:
```
python data/load_compas_data.py 1 false 1_FOLD_preprocessed
```
Then, run the following command to receive the results for three different potential decision maker utility functions:
```
python example_compas_PPV_FOR.py
```

#### Analyzing the utility fairness tradeoff

As we argue in our SDS'2022 paper, FPR parity can be a morally appropriate fairness criterion for the recidivism prediction case.
To analyze the tradeoff between the decision maker utility and fairness (as measured by the degree to which FPR parity is satisfied), first preprocess the data (generating 10 folds, just comparing Caucasian individuals with African-American individuals):
```
python data/load_compas_data.py 10 true 10_FOLDS_preprocessed
```
Then, evaluate the tradeoff and visualize the result (using a 10-fold cross-validation) by running:
```
python example_compas_utility_fairness_tradeoff.py
```

## References

1. [Enforcing Group Fairness in Algorithmic Decision Making: Utility Maximization Under Sufficiency](https://arxiv.org/abs/2206.02237) <br>
Joachim Baumann, Anikó Hannák, Christoph Heitz <br>
in *Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency*, ser. FAccT ’22. New York, NY, USA: Association for Computing Machinery, 2022. [https://doi.org/10.1145/3531146.3534645](https://doi.org/10.1145/3531146.3534645)
 
2. Group Fairness in Prediction-Based Decision Making: From Moral Assessment to Implementation <br>
Joachim Baumann, Christoph Heitz <br>
in *2022 9th Swiss Conference on Data Science (SDS) [forthcoming]*.
