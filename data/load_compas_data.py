# code from: https://github.com/algofairness/fairness-comparison

import pickle
import sys
import numpy
import numpy.random
import pandas as pd
from sklearn.linear_model import LogisticRegression as SKLearn_LR

numpy.random.seed(12345)

NUM_OF_FOLDS = 10
TRAINING_PERCENT = 2.0 / 3.0
DROP_NON_CAUCASIAN_NON_AFRICAN_AMERICAN = True
COMPAS_DATA_FILENAME = 'data/propublica-recidivism_numerical.csv'
sensitive = 'race'

FILENAME_ADDITION = "preprocessedAAAA"

if len(sys.argv) > 1:
    NUM_OF_FOLDS = int(sys.argv[1])
    DROP_NON_CAUCASIAN_NON_AFRICAN_AMERICAN = sys.argv[2].lower() == 'true'
    FILENAME_ADDITION = sys.argv[3]


class ProcessedCompasData():

    def __init__(self):
        self.sensitive_attrs = ['sex', 'race']
        self.df = pd.read_csv(COMPAS_DATA_FILENAME)
        self.splits = []
        self.has_splits = False
        if DROP_NON_CAUCASIAN_NON_AFRICAN_AMERICAN:
            races_to_keep = ["African-American", "Caucasian"]
            print("races_to_keep:", races_to_keep)
            print("races not to keep:", [
                  i for i in self.df.race.unique() if i not in races_to_keep])
            print(self.df.shape[0], "entries exist with all races.")
            print(sum(~self.df.race.isin(races_to_keep)), "entries are dropped.")
            print("This results in a total of", self.df[self.df.race.isin(
                races_to_keep)].shape[0], "entries.")
            self.df = self.df[self.df.race.isin(races_to_keep)]

    def create_train_test_splits(self, num):
        if self.has_splits:
            return self.splits
        for i in range(0, num):
            n = len(self.df)
            a = numpy.arange(n)
            numpy.random.shuffle(a)
            split_ix = int(n * TRAINING_PERCENT)
            train_fraction = a[:split_ix]
            test_fraction = a[split_ix:]
            train = self.df.iloc[train_fraction]
            test = self.df.iloc[test_fraction]
            self.splits.append((train, test))
        self.has_splits = True
        return self.splits

    def get_sensitive_attributes_with_joint(self):
        if len(self.sensitive_attrs) > 1:
            return self.sensitive_attrs + ['-'.join(self.sensitive_attrs)]
        return self.sensitive_attrs



def run_eval_alg(train, test, sensitive_attributes):
    # return data for one fold
    privileged_vals = ['Male', 'Caucasian', 'Male-Caucasian']
    class_attr = 'two_year_recid'

    # set index so that predictions can be easily mapped to ground truth again in post-processing
    train = train.set_index(pd.Index(list(range(0, len(train)))))
    test = test.set_index(pd.Index(list(range(0, len(test)))))

    # remove sensitive attributes from the training set
    train_df_nosensitive = train.drop(columns=sensitive_attributes)
    test_df_nosensitive = test.drop(columns=sensitive_attributes)

    # create and train the classifier
    classifier = SKLearn_LR(max_iter=1000)
    y = train_df_nosensitive[class_attr]
    X = train_df_nosensitive.drop(columns=class_attr)
    classifier.fit(X, y)

    # get the predictions on the test set
    X_test = test_df_nosensitive.drop(class_attr, axis=1)
    predictions = classifier.predict(X_test)

    predictions_on_train_set = classifier.predict(X)
    y_test = test_df_nosensitive[class_attr]
    predictions_proba_X_test = classifier.predict_proba(X_test)[:, 1]
    predictions_proba_X_train = classifier.predict_proba(X)[:, 1]
    my_dict = {
        "privileged_val": privileged_vals[1],
        "sensitive_attributes": sensitive_attributes,
        "train_df_sensitive_attrs": train[sensitive_attributes],
        "test_df_sensitive_attrs": test[sensitive_attributes],
        "X_train": X,
        "y_train": y,
        "X_test": X_test,
        "y_test": y_test,
        "predictions_proba_X_test": pd.Series(predictions_proba_X_test),
        "predictions_proba_X_train": pd.Series(predictions_proba_X_train),
        "predictions_X_test": pd.Series(predictions == 1),
        "predictions_X_train": pd.Series(predictions_on_train_set == 1)
    }
    return my_dict


def save_data_of_all_folds(data, num_folds):
    filename = "data/propublica-recidivism_" + FILENAME_ADDITION + ".pickle"
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def run(num_folds=NUM_OF_FOLDS):

    print("Evaluating dataset...")
    processed_dataset = ProcessedCompasData()
    train_test_splits = processed_dataset.create_train_test_splits(num_folds)
    sensitive_attributes = processed_dataset.get_sensitive_attributes_with_joint()
    print("Sensitive attribute: " + sensitive)

    if NUM_OF_FOLDS == 1:
        train, test = train_test_splits[0]
        data_all_folds = run_eval_alg(train, test, sensitive_attributes)
    else:
        data_all_folds = {}
        for i in range(0, num_folds):
            train, test = train_test_splits[i]
            try:
                data_all_folds[i] = run_eval_alg(train, test, sensitive_attributes)
            except Exception as e:
                import traceback
                traceback.print_exc(file=sys.stderr)
                print("Failed: %s" % e, file=sys.stderr)

    save_data_of_all_folds(data_all_folds, num_folds)


def main():
    run()


if __name__ == '__main__':
    main()
