

from collections import Counter
from gensim.models import word2vec 
#from glove import Glove 
#from glove import Corpus 
import argparse
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')




sentences = word2vec.Text8Corpus('text8')




model = word2vec.Word2Vec(sentences, size=80, sg=1, window=5, min_count=3, workers=4, alpha=0.02, seed=111, negative=6)




model['computer']



word_freq = Counter(model.wv.vocab.keys())
stopwords = ['the', 'of', 'and', 'a', 'to', 's']






model.similarity('boy', 'girl')




y1 = model.similarity("woman", "man")

print("--------\n")



y2 = model.most_similar("nice", topn=20)

for item in y2:
    print(item[0], item[1])
print("--------\n")


print(' "man" is to "father" as "woman" is to ...? \n')
y3 = model.most_similar(['woman', 'father'], ['man'], topn=3)
for item in y3:
    print(item[0], item[1])
print("--------\n")




more_examples = ["he his she", "big bigger bad", "going went being"]
for example in more_examples:
    a, b, x = example.split()
    predicted = model.most_similar([x, b], [a])[0][0]
    print("'%s' is to '%s' as '%s' is to '%s'" % (a, b, x, predicted))
print("--------\n")



model.save("text8.model")



from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train', 
                                  shuffle=True, random_state=11)
newsgroups_test = fetch_20newsgroups(subset='test', 
                                  shuffle=True, random_state=11)



print("\n".join(newsgroups_train.data[0].split("\n")))



import string
import re
def preProcessor(s):
    #s = s.encode('utf-8')
    s = re.sub('['+string.punctuation+']', ' ', s)
    s = re.sub('['+string.digits+']', ' ', s)
    s = re.sub('\n', ' ', s)
    s = s.lower()
    #s = s.translate(string.punctuation)
    return s


# In[25]:


preProcessor(newsgroups_train.data[0])


# In[26]:


from sklearn.feature_extraction.text import  TfidfVectorizer
#Obtain tf-idf vector for each article
#remove stopwords in Enlgish
tfidfVectorizer = TfidfVectorizer(stop_words='english', min_df=5, preprocessor=preProcessor, ngram_range=(1, 1))
X_train_tfidf = tfidfVectorizer.fit_transform(newsgroups_train.data)
X_train_tfidf.shape


# In[27]:


X_test_tfidf = tfidfVectorizer.transform(newsgroups_test.data)


# In[28]:


news_words = tfidfVectorizer.get_feature_names()
len(news_words)


# In[29]:


news_words[:20]



import time
#Train the model
from sklearn.naive_bayes import MultinomialNB
start = time.time()
clf_nb = MultinomialNB(0.1).fit(X_train_tfidf, newsgroups_train.target)
#Test the model
predicted = clf_nb.predict(X_test_tfidf)
end = time.time()
print('Accuracy of Naive Bayes: {:.3f}'.format(np.mean(predicted == newsgroups_test.target)))
print("Training and testing time (secs): {:.3f}".format(end - start))


# In[31]:


from sklearn.svm import SVC
clf_svc = SVC(C=1)
clf_svc.fit(X_train_tfidf, newsgroups_train.target)
#Test the model
predicted = clf_nb.predict(X_test_tfidf)
print('Accuracy of Naive Bayes: {:.3f}'.format(np.mean(predicted == newsgroups_test.target)))


# In[32]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=50, alpha=1e-4,
                    solver='adam', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.01)

mlp.fit(X_train_tfidf,newsgroups_train.target)
mlp.score(X_test_tfidf, newsgroups_test.target)



news_words_selected = set(news_words).intersection(set(model.wv.vocab.keys()))


# In[34]:


len(news_words_selected)



col_indice = [tfidfVectorizer.vocabulary_.get(w) for w in news_words_selected] 
X_train_tfidf_selected = X_train_tfidf[:, col_indice]
X_test_tfidf_selected = X_test_tfidf[:, col_indice]



def buildDocVector(doc_tfidf, Word_model, size):
    vec = np.zeros((1, size))    
    count = 0
    for i, word in enumerate(list(news_words_selected)):
        try:
            if doc_tfidf[0, i] != 0:           
                vec += Word_model[word].reshape((1, size)) * doc_tfidf[0, i]
                count += 1
        except:
            print('Error', word, i)
            continue
    if count != 0:
        vec /= count
    return vec



X_train_vec = np.concatenate([buildDocVector(doc_tfidf.toarray(), model, 80)
               for doc_tfidf in X_train_tfidf_selected])


# In[38]:


X_train_vec.shape


# In[39]:


X_test_vec = np.concatenate([buildDocVector(doc_tfidf.toarray(), model, 80)
               for doc_tfidf in X_test_tfidf_selected])


# In[40]:


#from sklearn.preprocessing import scale
#X_train_vec = scale(X_train_vec)
#X_test_vec = scale(X_test_vec)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
svc_model = SVC(random_state=1)
rf_model = RandomForestClassifier(random_state=111)
rf_model.fit(X_train_vec, newsgroups_train.target)
predicted = rf_model.predict(X_test_vec)
print('Accuracy of Random Forest: {:.3f}'.format(np.mean(predicted == newsgroups_test.target)))


# In[41]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=100, alpha=1e-4,
                    solver='adam', verbose=10, tol=1e-7, random_state=1,
                    learning_rate_init=.02)

mlp.fit(X_train_vec,newsgroups_train.target)
mlp.score(X_test_vec, newsgroups_test.target)




# In[42]:


def cleanText(corpus):
    corpus = [preProcessor(z) for z in corpus]
    corpus = [z.lower().replace('\n','').split() for z in corpus]    
    return corpus
train_corpus = cleanText(newsgroups_train.data)   
test_corpus = cleanText(newsgroups_test.data)   


# In[43]:


news_w2v = word2vec.Word2Vec(size=128, sg=0, window=5, min_count=3, workers=4)
news_w2v.build_vocab(train_corpus)
start_alpha = 0.005
end_alpha = 0.005
for i in range(5):
    news_w2v.train(train_corpus, total_examples=news_w2v.corpus_count, epochs=news_w2v.iter,
                  start_alpha=start_alpha, end_alpha=end_alpha)
    start_alpha -= 0.001
    end_alpha = start_alpha


# In[44]:


tfidfVectorizer.vocabulary_.get('world')


# In[45]:



def buildDocVector(news_w2v, text, text_index, tfidf, size):
    vec = np.zeros((1, size))    
    count = 0
    for word in text:
        try:
            word_index = tfidfVectorizer.vocabulary_.get(word)
            w = tfidf[text_index, word_index]
            vec += news_w2v[word].reshape((1, size)) * np.exp(w)
            count += 1
        except:
            #print('Error', word)
            continue
    if count != 0:
        vec /= count
    return vec


# In[46]:


X_train_vec = np.concatenate([buildDocVector(news_w2v, z, i, X_train_tfidf, 128)
               for i, z in enumerate(train_corpus)])


# In[47]:


X_test_vec = np.concatenate([buildDocVector(news_w2v, z, i, X_test_tfidf, 128)
               for i, z in enumerate(test_corpus)])


# In[48]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=2, random_state=111)
lr.fit(X_train_vec, newsgroups_train.target)
predicted = lr.predict(X_test_vec)
print('Accuracy of Random Forest: {:.3f}'.format(np.mean(predicted == newsgroups_test.target)))


