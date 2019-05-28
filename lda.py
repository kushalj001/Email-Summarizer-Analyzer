# Author - Kushal

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
import pandas as pd

eng_df = pd.read_pickle('./eng_df')
nlp = spacy.load('en_core_web_sm')
cv = CountVectorizer(max_df=0.95,min_df=2,stop_words='english')
term_matrix = cv.fit_transform(eng_df['Cleaned Emails'])
print(term_matrix)

lda = LatentDirichletAllocation(n_components=5)
lda.fit(term_matrix)

len(lda.components_)
print(lda.components_.shape)

print(lda.components_)

print(len(lda.components_[0]))

topic = lda.components_[0]
top_words_indices = topic.argsort()[-10:]
for index in top_words_indices:
    print(cv.get_feature_names()[index])

topic_word_dict = {}
for index,topic in enumerate(lda.components_):
    words = [cv.get_feature_names()[i] for i in topic.argsort()[-10:]]
    topic_word_dict[index] = words
    print('Top words for topic {}'.format(index))
    print(words)
    print('-'*120)

'''

Top words for topic 0
['feedback', 'issue', 'don', 'time', 'taking', 'scanners', 'thank', 'card', 'scanning', 'contact']
------------------------------------------------------------------------------------------------------------------------
Top words for topic 1
['restore', 'contact', 'facebook', 'device', 'mail', 'address', 'cards', 'google', 'account', 'backup']
------------------------------------------------------------------------------------------------------------------------
Top words for topic 2
['attention', 'caused', 'stores', 'tesco', 'loyalty', 'thank', 'cards', 'digital', 'information', 'acceptance']
------------------------------------------------------------------------------------------------------------------------
Top words for topic 3
['suggestions', 'don', 'questions', 'time', 'thank', 'contact', 'app', 'feedback', 'cards', 'card']
------------------------------------------------------------------------------------------------------------------------
Top words for topic 4
['code', 'pin', 'notifications', 'access', 'app', 'lock', 'contact', 'settings', 'touch', 'id']

'''


topics = lda.transform(term_matrix)
eng_df['topic'] = topics.argmax(axis=1)

def assign_topics(row):
    topic = row['topic']
    words = topic_word_dict[topic]

    return words

eng_df['topic words'] = eng_df.apply(assign_topics,axis=1)
print(eng_df.head())


print(eng_df['Cleaned Emails'][4])
print('-'*120)
print(eng_df['topic'][4])
print('-'*120)
print(topic_word_dict[eng_df['topic'][4]])
print('-'*120)

'''
Thank you so much for reaching out and taking the time to contact us about this issue! Please excuse the delayed response.
I'm happy to inform you that you can already enlarge the front and back pictures of your cards simply by tapping on it once.
Your card pictures will then get enlarged as well as rotated. However, I will also suggest to our developers to make zooming already
in the "Notes" tab possible for future versions of  . I hope I was able to help you.
If you have any further questions, suggestions for improvements or general feedback, please don't hesitate to contact me again.
------------------------------------------------------------------------------------------------------------------------
3
------------------------------------------------------------------------------------------------------------------------
['suggestions', 'don', 'questions', 'time', 'thank', 'contact', 'app', 'feedback', 'cards', 'card']
'''



# Other information

info = list(email_df['Other Info'])
info[0].split('---')

text = info[10].split('---')
text = ' ,'.join([x for x in text])
print(text)

doc = nlp(text)

ents = []
if doc.ents:
    for ent in doc.ents:
        ents.append(ent.text)
        print(f'{ent.text:{30}}{ent.label_:{30}}{spacy.explain(ent.label_):{60}}')
else:
    print('No entities')
    pass

'''
Merci                         ORG                           Companies, agencies, institutions, etc.
Isabelle van Capelleveen      PERSON                        People, including fictional
Customer Support              PERSON                        People, including fictional
C-HUB / Hafenstraße           ORG                           Companies, agencies, institutions, etc.
25-27                         CARDINAL                      Numerals that do not fall under another type
'''
