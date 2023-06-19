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
# # Methods
#
# We divide automatic methods for automatic error detection into two categories which we dub **flagger** and **scorer**.
# Flagging means that methods gives a dichotomous, binary judgement whether the label for an instance is correct or erroneous.
# Scoring methods on the other hand give a percentage estimate on how likely it is that an annotation is erroneous.
# These correspond to unranked and ranked evaluation from information retrieval and similarly require different evaluation metrics.
#
# **Flagger:**
#
# | **Abbreviation** | **Method**           | **Text** | **Token** | **Span** | **Proposed by**                                            |
# |------------------|----------------------|----------|-----------|----------|------------------------------------------------------------|
# | CL               | Confident Learning   | ✓        | ✓         | ✓        | Northcutt (2021)                                           |
# | CS               | Curriculum Spotter   | ✓        |           |          | Amiri (2018)                                               |
# | DE               | Diverse Ensemble     | ✓        | ✓         | ✓        | Loftsson (2009)                                            |
# | IRT              | Item Response Theory | ✓        | ✓         | ✓        | Rodriguez (2021)                                           |
# | LA               | Label Aggregation    | ✓        | ✓         | ✓        | Amiri (2018)                                               |
# | LS               | Leitner Spotter      | ✓        |           |          | Amiri (2018)                                               |
# | PE               | Projection Ensemble  | ✓        | ✓         | ✓        | Reiss (2020)                                               |
# | RE               | Retag                | ✓        | ✓         | ✓        | van Halteren (2000)                                        |
# | VN               | Variation n-Grams    |          | ✓         | ✓        | Dickinson (2003)                                           |
#
# **Scorer:**
#
# | **Abbreviation** | **Method**                 | **Text** | **Token** | **Span** | **Proposed by**                                            |
# |------------------|----------------------------|----------|-----------|----------|------------------------------------------------------------|
# | BC               | Borda Count                | ✓        | ✓         | ✓        | Larson (2020)                                              |
# | CU               | Classification Uncertainty | ✓        | ✓         | ✓        | Hendrycks (2017)                                           |
# | DM               | Data Map Confidence        | ✓        | ✓         | ✓        | Swayamdipta (2020)                                         |
# | DU               | Dropout Uncertainty        | ✓        | ✓         | ✓        | Amiri (2018)                                               |
# | KNN              | k-Nearest Neighbor Entropy | ✓        | ✓         | ✓        | Grivas (2020)                                              |
# | LE               | Label Entropy              |          | ✓         | ✓        | Hollenstein (2016)                                         |
# | MD               | Mean Distance              | ✓        | ✓         | ✓        | Larson (2019)                                              |
# | PM               | Prediction Margin          | ✓        | ✓         | ✓        | Dligach (2011)                                             |
# | WD               | Weighted Discrepancy       |          | ✓         | ✓        | Hollenstein (2016)                                         |
#
# We further divide methods by their means of annotation error detection and describe how to use each.

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Tasks and data
#
# We focus on methods for text classification as well as token and span labeling, but our implementations should be easily adaptable to other tasks.
# We provide example datasets that can be used to test methods and understand the data formats used.

# + pycharm={"is_executing": true, "name": "#%%\n"}
# Some imports
import awkward as ak

# + pycharm={"name": "#%%\n"}
from nessie.dataloader import load_example_text_classification_data, load_example_token_labeling_data

text_data = load_example_text_classification_data().subset(100)

token_data = load_example_token_labeling_data().subset(100)
token_data_flat = token_data.flatten() # Most methods need flat and not nested inputs, therefore, we flatten results here

# +
# Span
from nessie.dataloader import load_example_span_classification_data
from seqeval.metrics.sequence_labeling import get_entities

span_data = load_example_span_classification_data().subset(100)
span_data_flat = span_data.flatten() # Most methods need flat and not nested inputs, therefore, we flatten results here
span_noisy_entities = [e[0] for e in get_entities(span_data.noisy_labels.tolist())]

# + [markdown] pycharm={"name": "#%% md\n"}
# Different tasks have a different form of their inputs and output dimensions. For instance, token and span labeling typically are given as ragged/nested arrays, that is arrays of a varying second dimension (because the number of tokens is different per sentence). These are flattened when passed to most general methods and unflattened to their original shape. For span labeling, we also work on span-level and not on (BIO) tag level, therefore, spans need to be extracted and outputs from models need to be aggregated from token to spans. We also need to align predictions of models with the given span labels, as they can differ due to different boundary predictions. `nessie` provides helper functions for these operations.

# + [markdown] pycharm={"name": "#%% md\n"}
# ##  Variation based
#
# Methods based on the variation principle leverage the observation that similar surface forms are often only annotated with one or a few distinct labels.
# If an instance is annotated with a different, rarer label, then it is more often than not an annotation error or an inconsistency.
# Variation based methods are relatively easy to implement and can be used in settings for which it is difficult to train a machine learning model, being it because of low-resource scenarios or a task that is difficult to train models on.
# Their main disadvantage though is that they need overlapping surface forms to perform well, which is not the case in settings like text classification or datasets with diverse instances.

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Variation n-grams
#
# For each instance, n-gram contexts of different sizes are collected and compared to others.
# If the label for an instance disagrees with labels from other instances in the same context, then it is considered an error.

# + pycharm={"name": "#%%\n"}
# Token classification
from nessie.detectors import VariationNGrams

detector = VariationNGrams()
flags = detector.score(sentences=token_data.sentences, tags=token_data.noisy_labels)

# + pycharm={"name": "#%%\n"}
# Span classification
from nessie.detectors import VariationNGramsSpan

detector = VariationNGramsSpan()
flagged_spans = detector.score(sentences=span_data.sentences, tags=span_data.noisy_labels)

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Model Based
#
# Machine learning models trained on the to-be-corrected dataset can be used to find annotation errors.
# Models in this context are usually trained via cross-validation and the respective holdout set is used to detect errors.
# After all folds have been used as holdout, the complete dataset is analyzed.
# The training itself is not part of the method and is not altered by it, in contrast to other methods like the ones based on training dynamics.
# Several ways have been devised for model-based annotation error detection, which are described in the following.
# As the name implies, m-based annotation detection methods need trained models to obtain predictions or probabilities.
# We already implemented the most common models for you to be ready to use.
# We provide the following models:
#
# **Text classification:**
#
# | Class name                | Description                                   |
# |---------------------------|-----------------------------------------------|
# | FastTextTextClassifier    | Fasttext                                      |
# | FlairTextClassifier       | Flair                                         |
# | LgbmTextClassifier        | LightGBM with handcrafted features            |
# | LgbmTextClassifier        | LightGBM with S-BERT features                 |
# | MaxEntTextClassifier      | Logistic Regression with handcrafted features |
# | MaxEntTextClassifier      | Logistic with S-BERT features                 |
# | TransformerTextClassifier | Transformers                                  |
#
# **Sequence Classification:**
#
# | Class name                | Description                   |
# |---------------------------|-------------------------------|
# | FlairSequenceTagger       | Flair                         |
# | CrfSequenceTagger         | CRF with handcrafted features |
# | MaxEntSequenceTagger      | Maxent sequence tagger        |
# | TransformerSequenceTagger | Transformer                   |
#
# You can add your own models by implementing the respective abstract class for TextClassifier or SequenceTagger.
# Models are typicall trained via cross-validation, for which we provide a helper class.

# + pycharm={"name": "#%%\n"}
from nessie.helper import CrossValidationHelper
from nessie.models.text import DummyTextClassifier
from nessie.models.tagging import DummySequenceTagger

num_splits = 3 # Usually 10 is a good number, we use 3 for simplicity

# Text
cv = CrossValidationHelper(n_splits=num_splits)
tc_result = cv.run(text_data.texts, text_data.noisy_labels, DummyTextClassifier())

# Token
cv = CrossValidationHelper(n_splits=num_splits)
tl_result = cv.run_for_ragged(token_data.sentences, token_data.noisy_labels, DummySequenceTagger())
tl_result_flat = tl_result.flatten() # Most methods need flat and not nested inputs, therefore, we flatten results here

# Span
from nessie.task_support.span_labeling import align_span_labeling_result
cv = CrossValidationHelper(n_splits=num_splits)
sl_result = cv.run_for_ragged(span_data.sentences, span_data.noisy_labels, DummySequenceTagger())
# We extract spans from BIO tags, align them with model predictions and aggregate token level probabilities to span level
sl_result = align_span_labeling_result(span_data.noisy_labels, sl_result)

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Retag
#
# A simple way to use a trained model for annotation error detection is to use model predictions directly; when disagreeing with the given labels to correct, instances are flagged as annotation errors.

# + pycharm={"name": "#%%\n"}
from nessie.detectors import Retag

detector = Retag()

# Text
flags = detector.score(text_data.noisy_labels, tc_result.predictions)

# Token
flags_flat = detector.score(token_data_flat.noisy_labels, tl_result_flat.predictions)
flags = ak.unflatten(flags_flat, token_data.sizes)

# Span
flags = detector.score(sl_result.labels, sl_result.predictions)

# + [markdown] pycharm={"name": "#%%\n"}
# ### Classification Uncertainty
#
# Probabilistic classification models assign probabilities which are typically higher for instances that are correctly labeled compared to erroneous ones. Therefore, the class probabilities of the noisy labels can be used to score these for being an annotation error.

# +
from nessie.detectors import ClassificationUncertainty

detector = ClassificationUncertainty()

# Text
scores_cu_text = detector.score(labels=text_data.noisy_labels, probabilities=tc_result.probabilities, le=tc_result.le)

# Token
scores_cu_token_flat = detector.score(labels=token_data_flat.noisy_labels, probabilities=tl_result_flat.probabilities, le=tl_result_flat.le)
scores_cu_token = ak.unflatten(scores_cu_token_flat, token_data.sizes)

# Span
scores_cu_span = detector.score(labels=sl_result.labels, probabilities=sl_result.probabilities, le=sl_result.le)
# -

# ### Prediction Margin
#
# Inspired by active learning,  *Prediction Margin* uses the probabilities of the two highest scoring labels for an instance.
# The resulting score is simply their difference.
# The intuition behind this is that samples with smaller margin are more likely to be an annotation error, since the smaller the decision margin is the more unsure the model was.

# +
from nessie.detectors import PredictionMargin

detector = PredictionMargin()

# Text
scores_pm_text = detector.score(labels=text_data.noisy_labels, probabilities=tc_result.probabilities, le=tc_result.le)

# Token
scores_pm_token_flat = detector.score(labels=token_data_flat.noisy_labels, probabilities=tl_result_flat.probabilities, le=tl_result_flat.le)
scores_pm_token = ak.unflatten(scores_pm_token_flat, token_data.sizes)

# Span
scores_pm_span = detector.score(labels=sl_result.labels, probabilities=sl_result.probabilities, le=sl_result.le)
# -

# ### Confident Learning
#
# This method estimates the joint distribution of noisy and true labels. A threshold is then learnt (the average self-confidence) and instances whose computed probability of having the correct label is below the respective threshold are flagged as erroneous.

# +
from nessie.detectors import ConfidentLearning

detector = ConfidentLearning()

# Text
flags = detector.score(labels=text_data.noisy_labels, probabilities=tc_result.probabilities, le=tc_result.le)

# Token
flags_flat = detector.score(labels=token_data_flat.noisy_labels, probabilities=tl_result_flat.probabilities, le=tl_result_flat.le)
flags = ak.unflatten(flags_flat, token_data.sizes)

# Span
flags = detector.score(labels=sl_result.labels, probabilities=sl_result.probabilities, le=sl_result.le)
# -

# ### Dropout Uncertainty
#
# This method uses Monte Carlo dropout, that is, dropout during inference over several runs with different seeds to estimate the uncertainty of an underlying model. There are different acquisition methods to compute uncertainty from the stochastic passes, we use entropy over runs.

# +
from nessie.detectors import DropoutUncertainty

detector = DropoutUncertainty()

# Text
scores_du_text = detector.score(tc_result.repeated_probabilities)

# Token
scores_du_token_flat = detector.score(token_data_flat.repeated_probabilities)
scores_du_token = ak.unflatten(scores_du_token_flat, token_data.sizes)

# Span
scores_du_span = detector.score(sl_result.repeated_probabilities)
# -

# ### Label Aggregation
#
# Given repeated predictions obtained via *Monte Carlo Dropout*, one can use aggregation techniques from crowdsourcing like Dawid-Skene or MACE to adjudicate the resulting repeated predictions.

# +
from nessie.detectors import LabelAggregation

detector = LabelAggregation()

# Text
flags = detector.score(labels=text_data.noisy_labels, repeated_probabilities=tc_result.repeated_probabilities, le=tc_result.le)

# Token
flags_flat = detector.score(labels=token_data_flat.noisy_labels, repeated_probabilities=tl_result_flat.repeated_probabilities, le=tl_result_flat.le)
flags = ak.unflatten(flags_flat, token_data.sizes)

# Span
flags = detector.score(labels=sl_result.labels, repeated_probabilities=sl_result.repeated_probabilities, le=sl_result.le)
# -

# ## Training Dynamics
#
# Methods based on training dynamics use information derived from how a model behaves during training and how predictions change over the course of its training. The assumption behind both of these methods is that instances that are perceived harder or misqualified more frequently are more often annotation errors than easier ones.
#
# *Curriculum Spotter* and *Leitner Spotter* require that the instances can be scheduled independently. This is  for instance not the case for sequence labeling, as the model trains on complete sentences and not individual tokens or span. Even if they have different difficulties, they would end up in the same batch nonetheless.
#
# The implementation requires access to information during training, which is solved via callbacks. As only transformers have this avaiable, we only implmenet training dynamic methods for transformers.

# ### Curriculum Spotter
#
# This trains a model via curriculum learning, where the network trains on easier instances during earlier epochs and is then gradually introduced to harder instances.
# Instances then are ranked by how hard they were perceived during training. 

# +
from nessie.detectors import CurriculumSpotter

detector = CurriculumSpotter(max_epochs=2)
scores_cs_text = detector.score(texts=text_data.texts, labels=text_data.noisy_labels)
# -

# ### Leitner Spotter
#
# This method adapts the idea of the Zettelkasten to model training.
# There, difficult instances are presented more often during training than easier ones.

# +
from nessie.detectors import LeitnerSpotter

detector = LeitnerSpotter(max_epochs=2)
scores_ls_text = detector.score(texts=text_data.texts, labels=text_data.noisy_labels)
# -

# ### Data Map Confidence
#
# This method uses the class probability for each instance's gold label across epochs as a measure of confidence.
# It has been shown that low confidence correlates well with an item having a wrong label.

# +
# Text

from nessie.detectors import DataMapConfidence
from nessie.models.text import TransformerTextClassifier

detector = DataMapConfidence(TransformerTextClassifier(max_epochs=2))
scores_dm_text = detector.score(text_data.texts, text_data.noisy_labels)

# +
from nessie.detectors import DataMapConfidence
from nessie.models.tagging import TransformerSequenceTagger

# Token
detector = DataMapConfidence(TransformerSequenceTagger(max_epochs=2), needs_flattening=True)
scores_dm_token = detector.score(token_data.sentences, token_data.noisy_labels)
scores_dm_token_flat = ak.flatten(scores_dm_token).to_numpy()

# +
from nessie.detectors import DataMapConfidence
from nessie.models.tagging import TransformerSequenceTagger
from nessie.task_support.span_labeling import aggregate_scores_to_spans

# Span
detector = DataMapConfidence(TransformerSequenceTagger(max_epochs=2), needs_flattening=True)
scores_dm_span = detector.score(span_data.sentences, span_data.noisy_labels)
scores_dm_span = aggregate_scores_to_spans(span_data.noisy_labels, scores_dm_span)
scores_dm_span = ak.flatten(scores_dm_span).to_numpy()
# -

# ## Vector Space Proximity
#
# Approaches of this kind leverage dense embeddings of tokens, spans, and texts into a vector space and use their distribution therein.
# The distance of an instance to semantically similar instances is expected to be smaller than the distance to semantically different ones.
# Embeddings are typically obtained by using BERT-type models  for tokens and spans or S-BERT for  sentences.

# +
# Prepare the embeddings
from nessie.models.featurizer import CachedSentenceTransformer, FlairTokenEmbeddingsWrapper
from flair.embeddings import TransformerWordEmbeddings

# Text
sentence_embedder = CachedSentenceTransformer()
sentence_embeddings = sentence_embedder.embed(text_data.texts)

# Token
token_embedder = FlairTokenEmbeddingsWrapper(TransformerWordEmbeddings())
token_embeddings = token_embedder.embed(token_data.sentences, flat=True)

# Span
from nessie.task_support.span_labeling import embed_spans

span_embeddings = embed_spans(span_data.sentences, span_data.noisy_labels, token_embedder)
# -

# ### Mean Distance
#
# This method computes the centroid of each class by averaging vector embeddings of the respective instances.
# Items are then scored by the distance from their embedding vector to their centroid.
# The underlying assumption is that semantically similar items should have the same label and be close together (and thereby to the mean embedding) in the vector space.

# +
from nessie.detectors import MeanDistance

detector = MeanDistance()

# Text
scores_md_text = detector.score(labels=text_data.noisy_labels, embedded_instances=sentence_embeddings)

# Token
scores_md_token_flat = detector.score(labels=token_data_flat.noisy_labels, embedded_instances=token_embeddings)
scores_md_token = ak.unflatten(scores_md_token_flat, token_data.sizes)

# Span
scores_md_span = detector.score(labels=span_noisy_entities, embedded_instances=ak.flatten(span_embeddings).to_numpy())
# -

# ### k-Nearest-Neighbor Entropy
#
# For this method, all instances are first embedded into a vector space.
# Then, for every instance to check, its *k* nearest neighbors based on Euclidean distance in the vector space are retrieved.
# Their distances to the item's embedding vector are then used to compute a distribution over labels via applying softmax.
# An instance's score is then the entropy of its distance distribution; if it is large, it indicates uncertainty, hinting at being mislabeled.

# +
from nessie.detectors import KnnEntropy

detector = KnnEntropy()

# Text
scores_knn_text = detector.score(labels=text_data.noisy_labels, embedded_instances=sentence_embeddings)

# Token
scores_knn_token_flat = detector.score(labels=token_data_flat.noisy_labels, embedded_instances=token_embeddings)
scores_knn_token = ak.unflatten(scores_knn_token_flat, token_data.sizes)

# Span
scores_knn_span = detector.score(labels=span_noisy_entities, embedded_instances=ak.flatten(span_embeddings).to_numpy()) 
# -

# ## Ensembling
#
# Ensembling methods combine the scores or predictions of several individual flagger or scorer to obtain better performance than the sum of their parts.

# +
from nessie.helper import CrossValidationHelper
from nessie.models.text import DummyTextClassifier
from nessie.models.tagging import DummySequenceTagger

num_splits = 3 # Usually 10 is a good number, we use 3 for simplicity
cv = CrossValidationHelper(n_splits=num_splits)    

# +
# Text

# Replace these with non-dummy models
models = [DummyTextClassifier(), DummyTextClassifier(), DummyTextClassifier()]
collected_tc_predictions = []
for model in models:
    result = cv.run(text_data.texts, text_data.noisy_labels, model)    
    collected_tc_predictions.append(result.predictions)

# +
# Token

# Replace these with non-dummy models
models = [DummySequenceTagger(), DummySequenceTagger(), DummySequenceTagger()]
collected_tl_predictions = []
for model in models:
    result = cv.run_for_ragged(token_data.sentences, token_data.noisy_labels, model)    
    collected_tl_predictions.append(result.flatten().predictions)

# +
# Span

# Replace these with non-dummy models
models = [DummySequenceTagger(), DummySequenceTagger(), DummySequenceTagger()]
collected_sl_predictions = []
for model in models:
    result = cv.run_for_ragged(span_data.sentences, span_data.noisy_labels, model)    
    result_aligned = align_span_labeling_result(span_data.noisy_labels, result)
    collected_sl_predictions.append(result_aligned.predictions)
# -

# ### Diverse Ensemble
#
# Instead of using a single prediction like *Retag* does, here, the predictions of several models are aggregated.
# If most of them disagree on the label for an instance, then it is likely to be an annotation error.

# +
from nessie.detectors import MajorityVotingEnsemble

detector = MajorityVotingEnsemble()

# Text
flags = detector.score(text_data.noisy_labels, collected_tc_predictions)

# Token
flags = detector.score(token_data_flat.noisy_labels, collected_tl_predictions)

# Span
flags = detector.score(span_noisy_entities, collected_sl_predictions)
# -

# ### Item Response Theory
#
# Item Response Theory is a mathematical framework to model relationships between measured responses of test  subjects (e.g. answers to questions in an exam) for an underlying, latent trait  (e.g. the overall grasp on the subject that is tested).
# It can also be used to estimate the discriminative power of an item, i.e. how well the response to a question can be used to distinguish between subjects of different ability.
# In the context of AED, test subjects are trained models, the observations are the predictions on the dataset and the latent trait is task performance.

# +
from nessie.detectors import ItemResponseTheoryFlagger

detector = ItemResponseTheoryFlagger(num_iters=5) # Use 10,000 in real code

# Text
flags = detector.score(text_data.noisy_labels, collected_tc_predictions)

# Token
flags = detector.score(token_data_flat.noisy_labels, collected_tl_predictions)

# Span
flags = detector.score(span_noisy_entities, collected_sl_predictions)
# -

# ### Projection Ensemble
#
# This method trains an ensemble of logistic regression models on different Gaussian projections of BERT embeddings.
# If most of them disagree on the label for an instance, then it is likely to be an annotation error.

# +
from nessie.detectors import MaxEntProjectionEnsemble

detector = MaxEntProjectionEnsemble(n_components=[32, 64], seeds=[42], max_iter=100) # Use the defaults in real code

# TODO: Write me
# -

# ### Borda Count
#
# Similarly to combining several flagger into an ensemble, rankings obtained from different scorer can be combined as well.
# Here we leverage Borda counts, a voting scheme that assigns points based on their ranking.
# For each scorer, given scores for *N* instances, the instance that is ranked the highest is given *N* points, the second-highest *N-1* and so on.
# The points assigned by different scorers are then summed up for each instance and form the aggregated ranking.
# From out experiments, it is best to use only a few and well performing scorer when aggregating them way.

# +
import numpy as np

scores_text = np.vstack([
    scores_cu_text, scores_pm_text, # scores_du_text, 
    scores_ls_text, scores_cs_text, scores_dm_text, scores_md_text, scores_knn_text
])

scores_token = np.vstack([
    scores_cu_token_flat, scores_pm_token_flat, # scores_du_token_flat, 
    scores_dm_token_flat, scores_md_token_flat, scores_knn_token_flat,
])

scores_span = np.vstack([
    scores_cu_span, scores_pm_span, # scores_du_span, 
    scores_dm_span, scores_md_span,scores_knn_span,
])



# +
from nessie.detectors import BordaCount

detector = BordaCount()

# Text
scores_bc_text = detector.score(scores_text)

# Token
scores_bc_token_flat = detector.score(scores_token)

# Span
scores_bc_span = detector.score(scores_span)
