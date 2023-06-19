import sys, time
def reporttime(oldtime,label):
    newtime=int(time.time())
    print(f'{label}: {newtime-oldtime} sec')
    return newtime

starttime=int(time.time())

fname=sys.argv[1] if len(sys.argv)>1 else "text_classification.tsv"
mname=sys.argv[2] if len(sys.argv)>2 else ""
nsplits=int(sys.argv[3]) if len(sys.argv)>3 else 3

from nessie.dataloader import load_text_classification_tsv
text_data = load_text_classification_tsv(fname)
print(text_data.num_instances,text_data.num_labels)

import nessie
from nessie.helper import CrossValidationHelper
cv = CrossValidationHelper(n_splits=nsplits)

if mname=="fasttext":
    from nessie.models.text import FastTextTextClassifier
    classifier = FastTextTextClassifier()
elif mname=="transformer":
    from nessie.models.text import TransformerTextClassifier
    classifier = TransformerTextClassifier(max_epochs=2, model_name = "ssharoff/genres") # 'xlm-roberta-base')
else:
    from nessie.models.text import DummyTextClassifier
    classifier = DummyTextClassifier()
print(type(classifier))
runtime=reporttime(starttime,'Loaded')

tc_result = cv.run(text_data.texts, text_data.noisy_labels, classifier)
runtime=reporttime(starttime,'Evaluated')

from nessie.detectors import ConfidentLearning
detector = ConfidentLearning()
try:
    scores_cl_text = detector.score(labels=text_data.noisy_labels, probabilities=tc_result.probabilities, le=tc_result.le)
    print(scores_cl_text)
except Exception as e:
    print(repr(e))
runtime=reporttime(runtime,'ConfidentLearning')

from nessie.detectors import LabelAggregation
detector = LabelAggregation()
try:
    scores_la_text = detector.score(labels=text_data.noisy_labels, repeated_probabilities=tc_result.repeated_probabilities, le=tc_result.le)
    print(scores_la_text)
except Exception as e:
    print(repr(e))
runtime=reporttime(runtime,'LabelAggregation')

detector = nessie.detectors.Retag()
try:
    scores_re_text = detector.score(text_data.noisy_labels, tc_result.predictions)
    print(scores_re_text)
except Exception as e:
    print(repr(e))
runtime=reporttime(runtime,'Retag')

detector = nessie.detectors.BordaCount()
try:
    scores_bc_text = detector.score(text_data.noisy_labels, tc_result.predictions)
    print(scores_bc_text)
except Exception as e:
    print(repr(e))
runtime=reporttime(runtime,'BordaCount')

if mname=="transformer":
    detector = nessie.detectors.CurriculumSpotter(max_epochs=2)
    try:
        scores_cs_text = detector.score(texts=text_data.texts, labels=text_data.noisy_labels)
        print(scores_csn_text)
    except Exception as e:
        print(repr(e))
    runtime=reporttime(runtime,'CurriculumSpotter')

    detector = nessie.detectors.LeitnerSpotter(max_epochs=2)
    try:
        scores_ls_text = detector.score(texts=text_data.texts, labels=text_data.noisy_labels)
        print(scores_ls_text)
    except Exception as e:
        print(repr(e))
    runtime=reporttime(runtime,'LeitnerSpotter')

detector = nessie.detectors.ClassificationUncertainty()
try:
    scores_cu_text = detector.score(labels=text_data.noisy_labels, probabilities=tc_result.probabilities, le=tc_result.le)
    print(scores_cu_text)
except Exception as e:
    print(repr(e))
runtime=reporttime(runtime,'ClassificationUncertainty')

if mname=="transformer":
    detector = nessie.detectors.DataMapConfidence(TransformerTextClassifier(max_epochs=2))
    try:
        scores_dm_text = detector.score(text_data.texts, text_data.noisy_labels)
        print(scores_dm_text)
    except Exception as e:
        print(repr(e))
    runtime=reporttime(runtime,'DataMapConfidence')

# detector = nessie.detectors.
# print(scores__text)
# runtime=reporttime(runtime,'')

