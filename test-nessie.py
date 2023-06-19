from nessie.helper import CrossValidationHelper
from nessie.models.text import TransformerTextClassifier
from nessie.dataloader import load_text_classification_tsv
import nessie
import time
def reporttime(oldtime,label):
    newtime=int(time.time())
    print(f'{Label}: {newtime-oldtime} sec')
    return newtime

starttime=int(time.time())

text_data = load_text_classification_tsv("text_classification.tsv")

cv = CrossValidationHelper(n_splits=10)

classifier = TransformerTextClassifier(max_epochs=2, model_name = "ssharoff/genres") # 'xlm-roberta-base')
tc_result = cv.run(text_data.texts, text_data.noisy_labels, classifier)
runtime=reporttime(starttime,'Loaded')

detector = nessie.detectors.ConfidentLearning()
scores_cl_text = detector.score(labels=text_data.noisy_labels, probabilities=tc_result.probabilities, le=tc_result.le)
print(scores_cl_text)
runtime=reporttime(runtime,'ConfidentLearning')

detector = nessie.detectors.LabelAggregation()
scores_la_text = detector.score(labels=text_data.noisy_labels, repeated_probabilities=tc_result.repeated_probabilities, le=tc_result.le)
print(scores_la_text)
runtime=reporttime(runtime,'LabelAggregation')

detector = nessie.detectors.Retag()
scores_re_text = detector.score(text_data.noisy_labels, tc_result.predictions)
print(scores_re_text)
runtime=reporttime(runtime,'Retag')

detector = nessie.detectors.BordaCount()
scores_bc_text = detector.score(text_data.noisy_labels, tc_result.predictions)
print(scores_bc_text)
runtime=reporttime(runtime,'BordaCount')

detector = nessie.detectors.CurriculumSpotter(max_epochs=2)
scores_cs_text = detector.score(texts=text_data.texts, labels=text_data.noisy_labels)
print(scores_csn_text)
runtime=reporttime(runtime,'CurriculumSpotter')

detector = nessie.detectors.LeitnerSpotter(max_epochs=2)
scores_ls_text = detector.score(texts=text_data.texts, labels=text_data.noisy_labels)
print(scores_ls_text)
runtime=reporttime(runtime,'LeitnerSpotter')

detector = nessie.detectors.ClassificationUncertainty()
scores_cu_text = detector.score(labels=text_data.noisy_labels, probabilities=tc_result.probabilities, le=tc_result.le)
print(scores_cu_text)
runtime=reporttime(runtime,'ClassificationUncertainty')

detector = nessie.detectors.DataMapConfidence(TransformerTextClassifier(max_epochs=2))
scores_dm_text = detector.score(text_data.texts, text_data.noisy_labels)
print(scores_dm_text)
runtime=reporttime(runtime,'DataMapConfidence')

# detector = nessie.detectors.
# print(scores__text)
# runtime=reporttime(runtime,'')

