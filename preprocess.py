# Author - Kushal
import pandas as pd
import numpy as np
from langdetect import detect,detect_langs
from iso639 import languages
from langdetect import DetectorFactory
import cucco
from cucco import Cucco
import nltk
from nltk.stem.snowball import SnowballStemmer

DetectorFactory.seed = 0 # to deal with non-determinism

data = pd.read_excel('sampledata.xlsx')
print(data.head())

# Understanding the structure of emails

x = data['Information'][4]
print(x.split('\n'))

# There are duplicate emails in the dataset.

print('Total Emails: ',len(data))
print('Unique Emails: ',data.nunique())

def remove_salutation():
    '''
       Removes salutation and other information not required for summarization by simply
       considering the length of the sentence.
    '''

    emails = list(data['Information'])
    other_info = []
    email_text = []

    for sample in emails:
        #print(sample)
        #print('---------------------------------------------------------------------')
        sample = sample.split('\n')
        n = len(sample)
        text = []
        info = []
        for i in range(1,n-1):
            if len(sample[i])>50:
                text.append(sample[i])
            elif sample[i] != '':
                info.append(sample[i])

        info_text = '---'.join([x for x in info])
        str_text = ''.join([x for x in text])

        other_info.append(info_text)
        email_text.append(str_text)

    return email_text,other_info


emails,other_info = remove_salutation()
email_set = list(set(emails))
info_set = list(set(other_info))


def get_all_languages():
    '''Gets all the languages used in the dataset.'''

    langs = []
    lang_names = []
    for email in email_set:
        lang = detect(email)
        if lang not in langs:
            langs.append(lang)
            name = languages.get(alpha2=lang).name
            lang_names.append(name)


    return lang_names,langs


lang_names,langs = get_all_languages()
print(langs)
print(lang_names)


email_df = pd.DataFrame({'Cleaned Emails':emails,'Other Info':other_info})
print(email_df.head())


def get_lang(row):
    text = row['Cleaned Emails']
    lang = detect(text)
    return lang


email_df['Language'] = email_df.apply(get_lang,axis=1)
print(email_df['Language'].value_counts())


print(email_df.head())
indices = email_df[(email_df.Language=='ru')|(email_df.Language=='ja')|(email_df.Language=='pl')].index
email_df.drop(indices,inplace=True)


en_stemmer = SnowballStemmer('english')
fr_stemmer = SnowballStemmer('french')
de_stemmer = SnowballStemmer('german')
it_stemmer = SnowballStemmer('italian')
es_stemmer = SnowballStemmer('spanish')
nl_stemmer = SnowballStemmer('dutch')
stemmers = {'en':en_stemmer,'fr':fr_stemmer,'de':de_stemmer,'it':it_stemmer,'es':es_stemmer,'nl':nl_stemmer}

def stem_text(row):
    ''' Stems text based with snowball stemmer based on the language.'''

    lang = row['Language']
    text = row['Cleaned Emails']
    text = ''.join([x.lower() for x in text])
    #print(text)
    tokens = nltk.word_tokenize(text)
    #print(tokens)
    stemmer = stemmers[lang]
    #print(stemmer)
    stemmed_text = ' '.join([stemmer.stem(token) for token in tokens])

    return stemmed_text


email_df['Cleaned Emails'] = email_df.apply(stem_text,axis=1)
print(email_df['Cleaned Emails'][0])



norm_en = Cucco(language='en')
norm_es = Cucco(language='es')
norm_fr = Cucco(language='fr')
norm_it = Cucco(language='it')
norm_de = Cucco(language='de')
norm_nl = Cucco(language='nl')

normalisers = {'en':norm_en,'es':norm_es,'fr':norm_fr,'it':norm_it,'nl':norm_nl,'de':norm_de}

def normalise(row):
    ''' Performs text normalisation for multiple languages. Removes stopwords,punctuation etc.'''

    lang = row['Language']
    text = row['Cleaned Emails']
    sents = nltk.sent_tokenize(text)
    normaliser = normalisers[lang]
    rules = ['remove_stop_words', 'replace_punctuation', 'remove_extra_whitespaces']
    norm_text = ' '.join([normaliser.normalize(sent,rules) for sent in sents])

    return norm_text


email_df['Cleaned Emails'] = email_df.apply(normalise,axis=1)

print(email_df[email_df.Language=='fr']['Cleaned Emails'][10])
print(email_df[email_df.Language=='de']['Cleaned Emails'][17])
