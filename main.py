# approach this file with a bit of caution as i have some ai-assisted functions here

# python imports
import json
import os
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

# math imports and k-fold
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold

from tqdm.auto import tqdm

# local imports
from datasets import DEPENDENT, INDEPENDENT, hard_categories, memberships, sample
from fuzzy_zheng_tang import FuzzyZhengTang
from naive_bayes import NaiveBayes
from naive_credal import NaiveCredal
from naive_possibilistic import NaivePossibilistic
from tan import TAN
from tanc import TANC


SEEDS = range(5) # we used to have 3 seeds (5, 55, 555), but doing a range gives sexier and narrower error bars on plots

SYNTHETIC = {'Synthetic independent': INDEPENDENT, 'Synthetic dependent': DEPENDENT}
SYNTHETIC_TRAIN_SIZES = (30, 120, 480)
SYNTHETIC_TEST_SIZE = 2000

IRIS_TRAIN_SIZES = (30, 60, 120)
IRIS_FOLDS = 5

TIMING_REPEATS = 30 # good number for my pc


CONTROL = 'NB(1)' # NB(1) and TAN have same prior strenght, we keep it as a control so the tables produced leave it out
CREDAL = ('NCC', 'TANC')

METRICS = ['accuracy', 'singleton_rate', 'singleton_accuracy', 'coverage', 'mean_size', 'fit_ms', 'predict_ms',
           'predict_us', 'probability_mae', 'zero_score_rate', 'accuracy_on_NCC_singletons',
           'accuracy_on_TANC_singletons']


selected_classifiers = {'NB': NaiveBayes(), 'NB(1)': NaiveBayes(1), 'TAN': TAN(), 'NPC': NaivePossibilistic(), 'FNB': FuzzyZhengTang(), 'NCC': NaiveCredal(), 'TANC': TANC()}



def time_function(function, repeats=TIMING_REPEATS):
    function()  # run the called function
    start = perf_counter()

    for _ in range(repeats): result = function()
    return result, (perf_counter() - start) * 1000 / repeats


def run_split(dataset, X_train, y_train, X_test, y_test, centers, seed, fold, posterior=None):
    """
    fit, time, and eval all classifiers on one split
    """

    train_grades, test_grades = memberships(X_train, centers), memberships(X_test, centers)
    train_hard, test_hard = hard_categories(train_grades), hard_categories(test_grades)
    
    sizes = [len(c) for c in centers]
    observations = np.arange(len(y_test))
    rows, predictions = [], {}

    # rotate the classifiers btw splits -> decreases the chance of fixed timing order effect
    models = list(selected_classifiers.items())
    offset = (seed + fold) % len(models)

    for name, model in models[offset:] + models[:offset]:

        # we need to treat the data passed onto the models differently (grades, hard)
        if isinstance(model, FuzzyZhengTang):  
            fit, test = (lambda: model.fit(train_grades, y_train)), test_grades
        
        else:
            fit, test = (lambda: model.fit(train_hard, y_train, sizes)), test_hard

        # time via repeats here
        _, fit_ms = time_function(fit)

        # results out here
        predicted, predict_ms = time_function(lambda: model.predict(test))
        predictions[name] = predicted

        size = predicted.sum(axis=1)
        included = predicted[observations, y_test]
        singleton = size == 1
        
        credal = name in CREDAL  # one of the credal ones or not

        # expected accuracy of a classifier with unif. tie break rule (a bit of ai here)
        point_correct = np.full(len(y_test), np.nan) if credal else included / size


        row = {'dataset': dataset, 'seed': seed, 'fold': fold, 'n_train': len(y_train), 'classifier': name}
       
        row['accuracy'] = point_correct.mean()
        row['singleton_rate'] = singleton.mean()
        row['singleton_accuracy'] = included[singleton].mean() if singleton.any() else np.nan
       
        row['coverage'] = included.mean()
        row['mean_size'] = size.mean()
       
        row['fit_ms'] = fit_ms
        row['predict_ms'] = predict_ms
        row['predict_us'] = predict_ms * 1000 / len(y_test)
       
        row['probability_mae'] = np.nan
        row['zero_score_rate'] = np.nan if credal else np.mean(model.scores(test).sum(axis=1) == 0)

        if posterior is not None and isinstance(model, (NaiveBayes, TAN, FuzzyZhengTang)):
            scores = model.scores(test)
            total = scores.sum(axis=1, keepdims=True)

            # ai wrote this:
            # A row with only zero scores has no posterior. The error of the split is then undefined.
            if np.all(total > 0):
                row['probability_mae'] = np.abs(scores / total - posterior).mean()
            #  ------

        row['n_test'] = len(y_test)
        row['n_singletons'] = int(singleton.sum())
        row['n_singleton_correct'] = int(included[singleton].sum())
        row['n_covered'] = int(included.sum())

        row['set_size_total'] = int(size.sum())
        row['point_correct_expected'] = point_correct.sum()

        rows.append(row)

    # eval each point classifier on exactly the observations where a credal classifier returns one label.
    for row in rows:
        predicted = predictions[row['classifier']]
        point_correct = predicted[observations, y_test] / predicted.sum(axis=1)

        for reference in CREDAL:  # ofc treat the credals differently, and have to mask them
            mask = predictions[reference].sum(axis=1) == 1
            key = f'accuracy_on_{reference}_singletons'
            evaluate = mask.any() and row['classifier'] not in CREDAL

            # print(f"{mask=}")
            # print(f"{evaluate=}")

            row[key] = point_correct[mask].mean() if evaluate else np.nan
            row[key + '_n'] = int(mask.sum())
            row[key + '_correct'] = point_correct[mask].sum() if evaluate else np.nan

    return rows


def run_experiments():
    """
    run all with generated and iris data/splits
    """

    rows = []

    for dataset, table in tqdm(SYNTHETIC.items(), desc="fitting to generated datasets"):
        for seed in tqdm(SEEDS, desc="seeds", leave=False): 

            rng = np.random.default_rng(seed)

            X, y, _ = sample(table, max(SYNTHETIC_TRAIN_SIZES), rng)
            X_test, y_test, posterior = sample(table, SYNTHETIC_TEST_SIZE, rng)

            centers = [np.array([0., 1.]), np.array([0., 1.])]

            for n in SYNTHETIC_TRAIN_SIZES:
                rows += run_split(dataset, X[:n], y[:n], X_test, y_test, centers, seed, 0, posterior)

    X, y = load_iris(return_X_y=True)

    for seed in tqdm(SEEDS, desc="fitting to iris"):  
        folds = StratifiedKFold(IRIS_FOLDS, shuffle=True, random_state=seed)

        for fold, (train, test) in tqdm(enumerate(folds.split(X, y)), total=IRIS_FOLDS, desc="folds", leave=False):
            
            rng = np.random.default_rng(seed)
            per_class = [rng.permutation(train[y[train] == c]) for c in range(3)]

            for n in IRIS_TRAIN_SIZES:
                # balanced and nested subsets, with n / 3 observations of each class
                selected = np.concatenate([indices[:n // 3] for indices in per_class])

                centers = [np.linspace(column.min(), column.max(), 3) for column in X[selected].T]

                rows += run_split('Iris', X[selected], y[selected], X[test], y[test], centers, seed, fold)

    return pd.DataFrame(rows)


def summarize(runs):  # credal and mae parts are written by ai here
    seed_groups = runs.groupby(['dataset', 'n_train', 'classifier', 'seed'])
    per_seed = seed_groups[METRICS].mean()
    counts = seed_groups.sum(numeric_only=True, min_count=1)

    per_seed['accuracy'] = counts['point_correct_expected'] / counts['n_test']
    per_seed['singleton_rate'] = counts['n_singletons'] / counts['n_test']
    per_seed['singleton_accuracy'] = counts['n_singleton_correct'] / counts['n_singletons'].replace(0, np.nan)
    per_seed['coverage'] = counts['n_covered'] / counts['n_test']
    per_seed['mean_size'] = counts['set_size_total'] / counts['n_test']

    for reference in CREDAL:
        key = f'accuracy_on_{reference}_singletons'
        per_seed[key] = counts[key + '_correct'] / counts[key + '_n'].replace(0, np.nan)

    per_seed['probability_mae'] = seed_groups['probability_mae'].agg(lambda s: s.mean(skipna=False))

    groups = per_seed.groupby(['dataset', 'n_train', 'classifier'])
    means, sds, ses = groups.mean(), groups.std(), groups.sem()
    means['probability_mae'] = groups['probability_mae'].agg(lambda s: s.mean(skipna=False))

    return means.join(sds.add_suffix('_sd')).join(ses.add_suffix('_se'))


def main():
    runs = run_experiments()
    summary = summarize(runs)

    columns = [metric + suffix for metric in ('fit_ms', 'predict_ms', 'predict_us') for suffix in ('', '_sd', '_se')]

    timings = summary[columns].copy()
    timings.insert(0, 'n_test', runs.groupby(['dataset', 'n_train', 'classifier'])['n_test'].first())
    timings = timings.drop(CONTROL, level='classifier')

    metadata = {
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'logical_processors': os.cpu_count(),
        'python': platform.python_version(),
        'packages': {name: version(name) for name in ('numpy', 'pandas', 'scikit-learn')},
        'timer': 'time.perf_counter',
        'warmup_calls': 1,
        'timed_calls_per_operation': TIMING_REPEATS,
        'seeds': list(SEEDS),
        'synthetic_train_sizes': list(SYNTHETIC_TRAIN_SIZES),
        'synthetic_test_size': SYNTHETIC_TEST_SIZE,
        'iris_train_sizes': list(IRIS_TRAIN_SIZES),
        'iris_folds': IRIS_FOLDS,
        'classifiers': [name for name in selected_classifiers if name != CONTROL],
        'control': CONTROL,
        # 'timing_exclusions': ['Data generation', 'Preprocessing', 'Model construction', 'Imports', 'Evaluation metrics'],
    }

    output = Path(__file__).resolve().parent / 'results'
    output.mkdir(exist_ok=True)

    runs.to_csv(output / 'runs.csv', index=False)
    summary.to_csv(output / 'summary.csv')
    timings.to_csv(output / 'timings.csv')
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')

    print(f'see {output}')


if __name__ == '__main__':
    main()
