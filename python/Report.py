get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('white')
sns.set_palette('colorblind')

from imp import reload
import read_annotations
reload(read_annotations);

import show_metrics
reload(show_metrics);

GOLD_STANDARD = get_ipython().getoutput('echo "$GOLD_STANDARD_FILENAME"')
ANNOTATION_FILE = get_ipython().getoutput('echo "$ANNOTION_FILENAME"')
if GOLD_STANDARD[0] == '':
    GOLD_STANDARD = '/home/mteruel/am/data/echr/annotation/C/CASE_OF__ALKASI_v._TURKEY.txt'
else:
    GOLD_STANDARD = GOLD_STANDARD[0]
if ANNOTATION_FILE[0] == '':
    ANNOTATION_FILE = '/home/mteruel/am/data/echr/annotation/L/CASE_OF__ALKASI_v._TURKEY.txt'
else:
    ANNOTATION_FILE = ANNOTATION_FILE[0]

documents = read_annotations.read_parallel_annotations([('ann1', ANNOTATION_FILE), ('gold', GOLD_STANDARD)])
annotation = documents['ann1']
gold = documents['gold']

labels1, labels2 = read_annotations.get_labels(gold, annotation)
show_metrics.show_kappa(labels1, labels2, gold.identifier, annotation.identifier)
show_metrics.show_confusion_matrix(labels1, labels2, gold.identifier, annotation.identifier)

def get_simplified_labels(labels):
    return [x if x != 'major-claim' else 'claim' for x in labels]
labels1, labels2 = read_annotations.get_labels(gold, annotation)
labels1 = get_simplified_labels(labels1)
labels2 = get_simplified_labels(labels2)
show_metrics.show_kappa(labels1, labels2, gold.identifier, annotation.identifier)
show_metrics.show_confusion_matrix(labels1, labels2, gold.identifier, annotation.identifier)

STYLES = {'claim': 'red', 'premise': 'green', 'major-claim': 'blue'}

def similar_components(doc1, doc2, name1='A', name2='G', tolerance=0.5):
    """Samples sentences where the components share al least tolerance words.
    """
    for sentence1, sentence2 in zip(doc1.sentences, doc2.sentences):
        mismatching_words = sum(1 for i, j in zip(sentence1.labels, sentence2.labels) if i != j)
        labeled_words = max(
            len([i for i in sentence1.labels if i != doc1.default_label]),
            len([i for i in sentence2.labels if i != doc2.default_label]))
        if mismatching_words == 0 or mismatching_words > labeled_words * tolerance:
            continue
        # Print both sentences
        print(name1 + ': ' + sentence1.pretty_print(styles=STYLES))
        print(name2 + ': ' + sentence2.pretty_print(styles=STYLES))
        print('---')

similar_components(annotation, gold)

def unlabeled_components(doc1, doc2, name1='A', name2='G'):
    """Samples sentences where the components in doc2 are not labeled in doc1 or viceversa."""
    for sentence1, sentence2 in zip(doc1.sentences, doc2.sentences):
        mismatching_words = sum(1 for i, j in zip(sentence1.labels, sentence2.labels)
                                if i != j and (i == doc1.default_label or j == doc2.default_label))
        matching_words = sum(1 for i, j in zip(sentence1.labels, sentence2.labels) if i == j)
        if mismatching_words == 0 or matching_words > 0:
            continue
        # Print both sentences
        print(name1 + ': ' + sentence1.pretty_print(styles=STYLES))
        print(name2 + ': ' + sentence2.pretty_print(styles=STYLES))
        print('---')

unlabeled_components(annotation, gold)

def misslabeled_components(doc1, doc2, name1='A', name2='G', tolerance=0.5):
    """Samples sentences where the component label in doc2 does not match with the label in doc1,
    but both are labeled."""
    for sentence1, sentence2 in zip(doc1.sentences, doc2.sentences):
        mismatching_words = sum(1 for i, j in zip(sentence1.labels, sentence2.labels)
                                if i != j and i != doc1.default_label and j != doc2.default_label)
        labeled_words = max(
            len([i for i in sentence1.labels if i != doc1.default_label]),
            len([i for i in sentence2.labels if i != doc2.default_label]))
        if mismatching_words == 0:
            continue
        # Print both sentences
        print(name1 + ': ' + sentence1.pretty_print(styles=STYLES))
        print(name2 + ': ' + sentence2.pretty_print(styles=STYLES))
        print('---')

misslabeled_components(annotation, gold)



