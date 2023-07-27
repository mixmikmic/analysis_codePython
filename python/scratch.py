from spacy.en import English
from spacy.symbols import nsubj, VERB
from pattern.en import conjugate, PAST, PRESENT, tenses, SINGULAR, PLURAL
from itertools import tee
import string
from HTMLParser import HTMLParser
from change_tense.change_tense import change_tense, get_subjects_of_verb

nlp = English()

text = u"Alice was beginning to get very tired of sitting by her sister on the bank and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, ‘and what is the use of a book,’ thought Alice ‘without pictures or conversations?’ So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly White Rabbit with pink eyes ran close by her."
print HTMLParser().unescape(text)

print(change_tense(text,"present"))

print(change_tense(text, "future"))

text = "White rabbits with pink eyes ran close by her."
change_tense(text, 'present')

text_plural_check = "Rabbits with white fur ran close by her."
change_tense(text_plural_check, 'present')

text_person_test = u'I never said she stole my money.'
change_tense(text_person_test, 'present')

doc = nlp(text_person_test)
token = [x for x in list(doc.sents)[0] if x.text == 'said'][0]
subjects = [x.text for x in get_subjects_of_verb(token)]
'I' in subjects

text2 = """Dr. Dichter's interest in community psychiatry began as a fourth year resident when he and a co-resident ran a psychiatric inpatient and outpatient program at Fort McCoy Wisconsin treating formally institutionalized chronically mentally ill Cuban refugees from the Mariel Boatlift.  He came to Philadelphia to provide short-term inpatient treatment, alleviating emergency room congestion.  There he first encountered the problems of homelessness and was particularly interested in the relationship between the homeless and their families.  Dr. Dichter has been the Director of an outpatient department and inpatient unit, as well as the Director of Family Therapy at AEMC.  His work with families focused on the impact of chronic mental illness on the family system.  He was the first Medical Director for a Medicaid Managed Care Organization and has consulted with SAMHSA, CMS and several states assisting them to monitor access and quality of care for their public patients.  He currently is the Medical Director for Pathways to Housing PA, where he has assists chronically homeless to maintain stable housing and recover from the ravages of mental illness and substance abuse."""
text2

change_tense_spaCy(text2,'future')

text_person_test

