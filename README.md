# Should we play golf today?
## Uncertainty representation and reasoning in six classifiers
**2AMU30 - Group 21** 

This repository contains the necessary components for our real experiments used in the paper. These are (1) dataset generation, (2) models, and (3) results. 

---

## Run the experiments

We suggest using `uv` to run the code:
```shell
uv venv
source .venv/bin/activate
uv sync --frozen
uv run main.py
```

Without `uv` this can be achieved via pip:
```shell
python3 -m venv .venv
source .venv/bin/activate
pip install .
python main.py
```


Then, run the notebook (`results.ipynb`) start-to-finish in order to view the results we use in the paper.

## Files


* **`naive_bayes.py`**: Naive Bayes, and the counting and scoring functions that the other classifiers use
* **`tan.py`**: Tree-augmented naive Bayes, and the tree learning step that TANC uses
* **`naive_possibilistic.py`**: (product, not min) naive possibilistic classifier (NPC)
* **`fuzzy_zheng_tang.py`**: Zheng–Tang fuzzy naive Bayes (FNB)
* **`naive_credal.py`**: Naive credal classifier, and the IDM bounds and dominance rule that TANC uses
* **`tanc.py`**: Tree-based credal classifier 
* **`datasets.py`**: The synthetic data generator, membership grades, and hard categories
* **`main.py`**: Experiments, timing, evaluation, and output
* **`results.ipynb`**: Figures and tables of the results

We also include a smoothed (s=1) NB (called NB(1)), this we use as a control within NB. it has the same smoothing strength s as TAN so that it becomes comparable.

---
## Results 
All results can be viewed from the `results.ipynb` file. The release to GitHub includes the results of individual cells. When running locally, please see the running steps above.

Metadata for reproducibility:

```json
{
  "completed_at_utc": "2026-10-16T15:15:58.033776+00:00",
  "platform": "Windows-11-10.0.26200-SP0",
  "processor": "Intel64 Family 6 Model 186 Stepping 2, GenuineIntel",
  "logical_processors": 20,
  "python": "3.14.4",
  "packages": {
    "numpy": "2.5.3",
    "pandas": "3.0.6",
    "scikit-learn": "1.9.1"
  },
```

---
## Use of AI statement
When building this repository, we used AI to:
- Help with paper equation $\rightarrow$ numpy translations (especially for TANC and NCC)
- Double-check and suggest fixes for the model implementations
- Writing the table operations in `results.ipynb` due to time limitations
- in `main.py`: 
    - assistance in writing expected accuracy of a classifier with uniform tie break rule
    - zero-score posterior handling 
    - summarization of values (especially credal)