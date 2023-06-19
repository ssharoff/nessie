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
# # Datasets and tasks
#
# We focus on methods for text classification as well as token and span labeling, but our implementations should be easily adaptable to other tasks.
# We define simple file formats for these and provide loader for each.
# Also, we provide example datasets that can be used to test methods and understand the data formats used.

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Text classification

# + [markdown] pycharm={"name": "#%% md\n"}
# The goal of text classification is to assign a predefined category to a given text sequence  (which can for instance be a sentence, paragraph, or a document).
# Example applications are news categorization, sentiment analysis or intent detection.
# For text classification, we consider each individual sentence or document its own instance.

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Format
#
# The format consists of `n` rows with three tab-seperated fields, one for each instance.
# First is the text, then the gold label, finally the noisy label.

# + pycharm={"name": "#%%\n"}
# %%writefile text_classification.tsv
I love reindeer very much	positive	positive
I like Michael very much	positive	negative

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Example data

# + pycharm={"name": "#%%\n"}
from nessie.dataloader import load_example_text_classification_data

data = load_example_text_classification_data()
print(data[42])

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Data loader

# + pycharm={"name": "#%%\n"}
from nessie.dataloader import load_text_classification_tsv

ds = load_text_classification_tsv("text_classification.tsv")
print(ds[1])

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Token labeling
#
# The task of token labeling is to assign a label to each token.
# The most common task in this category is POS tagging.
# As there are not many other tasks with easily obtainable datasets, we only use two different POS tagging datasets.
# For token labeling, each individual token is considered an instance.

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Format
#
# The format is similar to other CoNLL formats. It consists of `n` blocks seperated by a blank line, one per sentence.
# Each block consists of a varying number of rows.
# Each row consists of three tab-seperated fields, one for each instance.
# First is the text, then the gold label, finally the noisy label.

# + pycharm={"name": "#%%\n"}
# %%writefile token_labeling.conll
I	PRON	PRON
like	VERB	NOUN
reindeer	NOUN	NOUN
.	PUNCT	PUNCT

I	PRON	PRON
adore	VERB	NOUN
Michael	PROPN	ADJ
very	ADV	ADV
much	ADV	VERB
.	PUNCT	PUNCT

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Example data

# + pycharm={"name": "#%%\n"}
from nessie.dataloader import load_example_token_labeling_data

data = load_example_token_labeling_data()
for token, gold_label, noisy_label in zip(*data[23]):
    print(token, gold_label, noisy_label)

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Data loader

# + pycharm={"name": "#%%\n"}
from nessie.dataloader import load_sequence_labeling_dataset

data = load_sequence_labeling_dataset("token_labeling.conll")

for token, gold_label, noisy_label in zip(*data[1]):
    print(token, gold_label, noisy_label)

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Span labeling
#
# Span labeling assigns labels not to single tokens, but to spans of text.
# Common tasks that can be modeled that way are named-entity recognition (NER), slot filling or chunking.

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Format
#
# The format is similar to other CoNLL formats and the same as for token labeling.
# It consists of `n` blocks seperated by a blank line, one per sentence.
# Each block consists of a varying number of rows.
# Each row consists of three tab-seperated fields, one for each instance.
# First is the text, then the gold label, finally the noisy label.

# + pycharm={"name": "#%%\n"}
# %%writefile span_labeling.conll
The	O	O
United	B-LOC	B-LOC
States	I-LOC	I-LOC
of	I-LOC	I-LOC
America	I-LOC	I-LOC
is	O	O
in	O	O
the	O	O
city	O	O
of	O	O
New	B-LOC	B-PER
York	I-LOC	I-PER
.	O	O

Hogwarts	B-ORG	B-ORG
pays	O	O
taxes	O	O
in	O	O
Manhattan	B-LOC	B-ORG
.	O	O

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Example data

# + pycharm={"name": "#%%\n"}
from nessie.dataloader import load_example_span_classification_data

data = load_example_span_classification_data()
for token, gold_label, noisy_label in zip(*data[80]):
    print(token, gold_label, noisy_label)

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Data loader

# + pycharm={"name": "#%%\n"}
from nessie.dataloader import load_sequence_labeling_dataset

data = load_sequence_labeling_dataset("span_labeling.conll")

for token, gold_label, noisy_label in zip(*data[1]):
    print(token, gold_label, noisy_label)
