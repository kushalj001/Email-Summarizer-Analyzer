# Author - Kushal

import pickle
import numpy as np
import pandas as pd
from langdetect import detect,detect_langs
from nltk import sent_tokenize
from bert_serving.client import BertClient
from sklearn.cluster import KMeans

df = pd.read_pickle('./eng_df')

print(df.head())
bert = BertClient()
def get_embeddings(row):
    '''Generates bert sentence embeddings.Bert server is open in the terminal.'''

    text = row['Cleaned Emails']
    sents = sent_tokenize(text)
    embeddings = bert.encode(sents)
    return embeddings

df['Embeddings'] = df.apply(get_embeddings,axis=1)

def get_cluster_centers(row):
    '''
    Performs clustering of sentences in the text. Number of clusters or the number of required sentences in summary
    is the square root of total sentences in the text.Returns cluster centers.
    '''

    text = row['Cleaned Emails']
    sents = sent_tokenize(text)
    clusters = int(np.ceil(len(sents)**0.5))
    embeddings = row['Embeddings']
    kmeans = KMeans(n_clusters=clusters).fit(embeddings)

    return kmeans.cluster_centers_

df['Cluster Centers'] = df.apply(get_cluster_centers,axis=1)

def get_summary(row):
    '''
    Generates summary by choosing the sentences in the text that are closest to the centroid.
    '''
    text = row['Cleaned Emails']
    sents = sent_tokenize(text)
    centroids = row['Cluster Centers']
    embeddings = row['Embeddings']
    clusters = centroids.shape[0]
    sents_len = len(sents)
    summary = []
    for i in range(clusters):
        select = -1
        m = -np.inf
        for j in range(sents_len):
            similarity = np.dot(centroids[i],embeddings[j])
            if similarity > m:
                m = similarity
                select = j
        summary.append(select)
    summary.sort()
    summary = ''.join([sents[i] for i in summary])
    return summary

df['Summary'] = df.apply(get_summary,axis=1)

print(df.head())

print(df['Cleaned Emails'][4])
print('-'*120)
print(df['Summary'][4])

'''
Thank you so much for reaching out and taking the time to contact us about this issue! Please excuse the delayed response.
I'm happy to inform you that you can already enlarge the front and back pictures of your cards simply by tapping on it once.
Your card pictures will then get enlarged as well as rotated. However, I will also suggest to our developers to make zooming already in
the "Notes" tab possible for future versions of . I hope I was able to help you. If you have any further questions, suggestions for
improvements or general feedback, please don't hesitate to contact me again.
------------------------------------------------------------------------------------------------------------------------
Thank you so much for reaching out and taking the time to contact us about this issue!However, I will also suggest to our developers to
make zooming already in the "Notes" tab possible for future versions of  .
If you have any further questions, suggestions for improvements or general feedback, please don't hesitate to contact me again.
'''


print(df['Cleaned Emails'][8])
print('-'*120)
print(df['Summary'][8])

'''
Thank you so much for reaching out and taking the time to send us feedback! We really appreciate it and are happy to hear that you
like our app!We are aware that the current loading times of our app have increased over the course of the latest updates and I can
assure you that our developers are already working on improving the speed and overall performance again for future releases. Until
 then, you could try keeping   opened in the background while shopping as a temporary workaround. This way, the app doesn't have
 to reload all your information (e.g. card pictures, points balances, etc.) completely each time it is opened and the loading times
 will be decreased significantly. In the meantime, I sincerely apologize for the inconvenience this causes and hope that you can use
 in its full capacity again soon.If you have any further questions, suggestions for improvements or general feedback, please don't
 hesitate to contact me again.
------------------------------------------------------------------------------------------------------------------------
We really appreciate it and are happy to hear that you like our app!We are aware that the current loading times of our app have
increased over the course of the latest updates and I can assure you that our developers are already working on improving the
speed and overall performance again for future releases.card pictures, points balances, etc.)completely each time it is opened
and the loading times will be decreased significantly.
'''
