# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] pycharm={"name": "#%% md\n"}
# # Evaluation
#
# In order to evaluate the performance of different methods and hyper-parameters, we leverage several metrics.
# As described before, we differentiate between two kinds of annotation error detectors, *flagger* and *scorer*.
# These need different metrics during evaluation, similar to classification and ranking or unranked and ranked evaluation from information retrieval.
# As flagging is a binary classification task, we use the standard metrics for this task which are precision, recall, and  F1.
# We also record the percentage of instances flagged .
# Scorer produce a ranking as seen in information retrieval.
# We use average precision (AP, also known as Area Under the Precision-Recall Curve (AUPR/AUPRC). In AED, AP is also identical to mean average precision (mAP) used in other works.) , Precision@10%, and Recall@10%.
# There are reasons why both precision and recall can be considered the more important metric of the two.
# A low precision leads to increased cost because many more instances than necessary need to be inspected manually after detection.
# Similarly, a low recall leads to problems because there still can be errors left after the application of AED.
# As both arguments have merit, we will mainly use the aggregated metrics F1 and AP.
# Precision and recall at 10% evaluate a scenario in which a scorer was applied and the first 10%  with the highest score (most likely to be wrongly annotated) are manually corrected.
#
# In contrast to other works, we explicitly do not use ROC AUC and discourage its use for AED, as it heavily overestimates performance when applied to imbalanced datasets.
# Datasets needing AED are typically very imbalanced because there are far more correct labels than wrong ones.

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Flaggers
#
# Evaluating flaggers is similar to evaluating classification.To evaluate flagger, we use precision, recall, F1, and % of instances flagged. 

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Text classification

# + pycharm={"name": "#%%\n"}
from sklearn.metrics import precision_recall_fscore_support

from nessie.detectors import Retag
from nessie.dataloader import load_example_text_classification_data
from nessie.helper import CrossValidationHelper
from nessie.models.text import FastTextTextClassifier
from nessie.metrics import percentage_flagged_score

ds = load_example_text_classification_data()

model = FastTextTextClassifier()
detector = Retag()

# Running AED
cv = CrossValidationHelper(n_splits=3)

cv_result = cv.run(ds.texts, ds.noisy_labels, model)
predicted_flags = detector.score(ds.noisy_labels, cv_result.predictions)

# Evaluation
precision, recall, f1, _ = precision_recall_fscore_support(ds.flags, predicted_flags, average="binary")
percent_flagged = percentage_flagged_score(ds.flags, predicted_flags)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
print(f"% flagged: {percent_flagged}")

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Token labeling
#
# In order to evaluate token labeling, we first flatten the ragged flags and then evaluate similarly to text classification.

# + pycharm={"name": "#%%\n"}
from sklearn.metrics import precision_recall_fscore_support

from nessie.detectors import Retag
from nessie.dataloader import load_example_token_labeling_data
from nessie.helper import CrossValidationHelper
from nessie.models.tagging import CrfSequenceTagger
from nessie.metrics import percentage_flagged_score

ds = load_example_token_labeling_data().subset(100)
ds_flat = ds.flatten()

model = CrfSequenceTagger()
detector = Retag()

# Running AED
cv = CrossValidationHelper(n_splits=3)

cv_result = cv. run_for_ragged(ds.sentences, ds.noisy_labels, model)
cv_result_flat = cv_result.flatten()

predicted_flags_flat = detector.score(ds_flat.noisy_labels, cv_result_flat.predictions)

# Evaluation
precision, recall, f1, _ = precision_recall_fscore_support(ds_flat.flags, predicted_flags_flat, average="binary")
percent_flagged = percentage_flagged_score(ds_flat.flags, predicted_flags_flat)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
print(f"% flagged: {percent_flagged}")

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Sequence labeling
#
# In order to evaluate sequence labeling, we align and aggregate predictions to have a list of spans and then evaluate similarly to text classification.

# + pycharm={"name": "#%%\n"}
from sklearn.metrics import precision_recall_fscore_support

from nessie.detectors import Retag
from nessie.dataloader import load_example_span_classification_data
from nessie.helper import CrossValidationHelper
from nessie.models.tagging import CrfSequenceTagger
from nessie.task_support.span_labeling import align_span_labeling_result, align_span_labeling_data

# Load and align data
ds = load_example_span_classification_data().subset(100)
aligned_data = align_span_labeling_data(ds.sentences, ds.gold_labels, ds.noisy_labels)

model = CrfSequenceTagger()
detector = Retag()

# Running AED
cv = CrossValidationHelper(n_splits=3)

cv_result = cv.run_for_ragged(ds.sentences, ds.noisy_labels, model)

# We extract spans from BIO tags, align them with model predictions and 
# aggregate token level probabilities to span level
cv_result_aligned = align_span_labeling_result(ds.noisy_labels, cv_result)

predicted_flags_aligned = detector.score(cv_result_aligned.labels, cv_result_aligned.predictions)

# Evaluation
precision, recall, f1, _ = precision_recall_fscore_support(aligned_data.flags, predicted_flags_aligned, average="binary")
percent_flagged = percentage_flagged_score(aligned_data.flags, predicted_flags_aligned)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
print(f"% flagged: {percent_flagged}")

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Scorers
#
# Evaluating scorers is similar to evaluating ranked retrieval in Information Retrieval. To evaluate flagger, we use precision@10, recall@10, and average precision. 

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Text classification

# + pycharm={"name": "#%%\n"}
import ireval

from nessie.detectors import ClassificationUncertainty
from nessie.dataloader import load_example_text_classification_data
from nessie.helper import CrossValidationHelper
from nessie.models.text import FastTextTextClassifier

ds = load_example_text_classification_data()

model = FastTextTextClassifier()
detector = ClassificationUncertainty()

# Running AED
cv = CrossValidationHelper(n_splits=3)

cv_result = cv.run(ds.texts, ds.noisy_labels, model)
scores = detector.score(ds.noisy_labels, cv_result.probabilities, cv_result.le)

# Evaluation
precision_at_10_percent = ireval.precision_at_k_percent(ds.flags, scores, 10)
recall_at_10_percent = ireval.recall_at_k_percent(ds.flags, scores, 10)
ap = ireval.average_precision(ds.flags, scores)


print(f"Precision@10%: {precision_at_10_percent}")
print(f"Recall@10%: {recall_at_10_percent}")
print(f"Average precision: {ap}")

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Token labeling
#
# In order to evaluate token labeling, we first flatten the ragged flags and then evaluate similarly to text classification.

# + pycharm={"name": "#%%\n"}
import ireval

from nessie.detectors import ClassificationUncertainty
from nessie.dataloader import load_example_token_labeling_data
from nessie.helper import CrossValidationHelper
from nessie.models.tagging import CrfSequenceTagger

ds = load_example_token_labeling_data().subset(100)
ds_flat = ds.flatten()

model = CrfSequenceTagger()
detector = ClassificationUncertainty()

# Running AED
cv = CrossValidationHelper(n_splits=3)

cv_result = cv. run_for_ragged(ds.sentences, ds.noisy_labels, model)
cv_result_flat = cv_result.flatten()

scores_flat = detector.score(ds_flat.noisy_labels, cv_result_flat.probabilities, cv_result.le)

# Evaluation
precision_at_10_percent = ireval.precision_at_k_percent(ds_flat.flags, scores_flat, 10)
recall_at_10_percent = ireval.recall_at_k_percent(ds_flat.flags, scores_flat, 10)
ap = ireval.average_precision(ds_flat.flags, scores_flat)

print(f"Precision@10%: {precision_at_10_percent}")
print(f"Recall@10%: {recall_at_10_percent}")
print(f"Average precision: {ap}")

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Sequence labeling
#
# In order to evaluate sequence labeling, we align and aggregate predictions to have a list of spans and then evaluate similarly to text classification.

# + pycharm={"name": "#%%\n"}
import ireval

from nessie.detectors import ClassificationUncertainty
from nessie.dataloader import load_example_span_classification_data
from nessie.helper import CrossValidationHelper
from nessie.models.tagging import CrfSequenceTagger
from nessie.task_support.span_labeling import align_span_labeling_result, align_span_labeling_data

# Load and align data
ds = load_example_span_classification_data().subset(100)
aligned_data = align_span_labeling_data(ds.sentences, ds.gold_labels, ds.noisy_labels)

model = CrfSequenceTagger()
detector = ClassificationUncertainty()

# Running AED
cv = CrossValidationHelper(n_splits=3)

cv_result = cv.run_for_ragged(ds.sentences, ds.noisy_labels, model)

# We extract spans from BIO tags, align them with model predictions and 
# aggregate token level probabilities to span level
cv_result_aligned = align_span_labeling_result(ds.noisy_labels, cv_result)

scores_aligned = detector.score(cv_result_aligned.labels, cv_result_aligned.probabilities, cv_result_aligned.le)

# Evaluation
precision_at_10_percent = ireval.precision_at_k_percent(aligned_data.flags, scores_aligned, 10)
recall_at_10_percent = ireval.recall_at_k_percent(aligned_data.flags, scores_aligned, 10)
ap = ireval.average_precision(aligned_data.flags, scores_aligned)

print(f"Precision@10%: {precision_at_10_percent}")
print(f"Recall@10%: {recall_at_10_percent}")
print(f"Average precision: {ap}")
