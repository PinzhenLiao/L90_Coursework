
# coding: utf-8

# In previous projects, we use TfIdf vectors to represent texts. However, in this project, we intend to leverage topic models, that is to represent each text with a linear combination of topics, which are also vectors.
# 
# The scikit-learn version: 0.19.1
# In order to simplify the classification, instead of using 20 categories of news, we just relabeled them as 4 major categories.

# In[28]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
get_ipython().run_line_magic('matplotlib', 'inline')


# In[29]:


#Read data, those data only has for major categories
newsgroups_train = pd.read_csv('news_train.csv')
newsgroups_test = pd.read_csv('news_test.csv')


# In[31]:





# In[3]:


newsgroups_train.head()


# ## 1.Get TfIdf Vectors

# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidfVectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=50000)
X_train_tfidf = tfidfVectorizer.fit_transform(newsgroups_train.News)
X_train_tfidf.shape


# In[5]:


X_test_tfidf = tfidfVectorizer.transform(newsgroups_test.News)
X_test_tfidf.shape


# In[6]:


feature_words = tfidfVectorizer.get_feature_names()


# ## 2.Extract topics by NMF

# In[7]:


from sklearn.decomposition import NMF, LatentDirichletAllocation
from time import time


# ### NMF Extraction
# NMF stands for "Non-negative Matrix Factorization", it is a dimension reduction technique. It can also be viewed as a way to extract topics from original vectors.

# In[8]:


n_components = 10
# Fit the NMF model, suppose there are 10 topics
t0 = time()
nmf = NMF(n_components=n_components, random_state=1,
          alpha=.1, l1_ratio=.5)
nmf.fit(X_train_tfidf)
print("done in %0.3fs." % (time() - t0))


# In[9]:


print('The shape of the topic matrix:', nmf.components_.shape)
print(nmf.components_[0, :])


# We can regard each text as a combination of weighted topic, for some topics the weight are larger whereasw some are small, and those weights can be features to represent the text. And Each topic is a combination of weights of words. Let's check the first compoent, namely the first topic.

# In[10]:


for i, component in enumerate(nmf.components_):
    #The most important 20 words' index
    nlargest = component.argsort()[-10:]
    word_list = []
    for n in nlargest:
        word_list.append(feature_words[n])
    print('*'*20)
    print('The ' + str(i) + 'th component:')
    print(word_list)


# It looks like the last compoenent is related to computer science, whereas the second one is related to religion. Next, we can extract the features for each news.

# In[11]:


#Extract feature for each news
nmf_train_features = nmf.transform(X_train_tfidf)
nmf_test_features = nmf.transform(X_test_tfidf)


# In[12]:


nmf_train_features.shape


# In[13]:


nmf_train_features[0, :]


# ### Classify
# 
# Based on the extracted features above, we can move forward, use machine learning algorithms to classify those news.

# In[14]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
knn = KNeighborsClassifier(n_neighbors=20,weights='distance')
knn.fit(nmf_train_features, newsgroups_train.Type)
predicted = knn.predict(nmf_test_features)


# In[15]:


print('Micro F1: {:.3f}'.format(f1_score(newsgroups_test.Type, predicted, average='micro')))


# The result is not satisfying, in order to select proper parameters and proper models we need to put two steps together and make a pipeline.

# ### Pipeline

# In[16]:


from sklearn.pipeline import make_pipeline
#Load classification Models
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, alpha=1e-4,
                    solver='sgd', verbose=False, tol=1e-5, random_state=1,
                    learning_rate_init=.17)
nb = MultinomialNB(0.01)
lr = LogisticRegression(C=0.5)
rf =  RandomForestClassifier(n_estimators=100, n_jobs=4)
svc = SVC(C=0.5)
ld = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
models = [nb, lr, rf, svc, ld, mlp]


# In[17]:


def make_plot(training_scores, testing_scores, title):
    '''Make plots'''
    plt.title(title)
    plt.plot(training_scores, marker='.', label='Training Score')
    plt.plot(testing_scores, marker='.', label='Testing Score')
    plt.legend()
    plt.ylabel('f1_score')


# In[18]:


#Specify the number of topics
n_components = [5, 10, 15, 20, 25,30]
#Traverse each 
plt.figure(figsize=(10, 12))
for i, model in enumerate(models):
    train_scores = []
    test_scores = []
    print('Start to run', str(model.__class__))
    for n_component in n_components:
        nmf = NMF(n_components=n_component, random_state=1,
                  alpha=.1, l1_ratio=.5)
       #Make pipeline to simplify code
        pipeline = make_pipeline(nmf, model)
        pipeline.fit(X_train_tfidf, newsgroups_train.Type)
        train_pred = pipeline.predict(X_train_tfidf)
        test_pred = pipeline.predict(X_test_tfidf)
        test_scores.append(f1_score(newsgroups_test.Type, test_pred, average='micro'))
        train_scores.append(f1_score(newsgroups_train.Type, train_pred, average='micro'))
    plt.subplot(3, 2 , i+1)
    title = str(model.__class__).replace("'>", '').split('.')[-1]
    make_plot(train_scores, test_scores, title)  
    plt.axis((0, 6, 0, 1))
    plt.tight_layout() 


# ## 3.Extract topics by LDA

# In[19]:


lda = LatentDirichletAllocation(n_components =10, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda.fit(X_train_tfidf)


# In[20]:


#Check the most important words within each topic,namely each component
for i, component in enumerate(lda.components_):
    #The most important 20 words' index
    nlargest = component.argsort()[-10:]
    word_list = []
    for n in nlargest:
        word_list.append(feature_words[n])
    print('*'*20)
    print('The ' + str(i) + 'th component:')
    print(word_list)


# ### Make a pipeline
# 
# Next, we will put LDA and classification models together.

# In[21]:


#Specify the number of topics
n_components = [5, 10, 15, 20, 25,30]
#Traverse each 
plt.figure(figsize=(10, 12))
for i, model in enumerate(models):
    train_scores = []
    test_scores = []
    print('Start to run', str(model.__class__))
    for n_component in n_components:
        lda = LatentDirichletAllocation(n_components =n_component, max_iter=10,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
       #Make pipeline to simplify code
        pipeline = make_pipeline(lda, model)
        pipeline.fit(X_train_tfidf, newsgroups_train.Type)
        train_pred = pipeline.predict(X_train_tfidf)
        test_pred = pipeline.predict(X_test_tfidf)
        test_scores.append(f1_score(newsgroups_test.Type, test_pred, average='micro'))
        train_scores.append(f1_score(newsgroups_train.Type, train_pred, average='micro'))
    plt.subplot(3, 2 , i+1)
    title = str(model.__class__).replace("'>", '').split('.')[-1]
    make_plot(train_scores, test_scores, title) 
    plt.axis((0, 6, 0, 1))
    plt.tight_layout() 


# It seems the topic models do not work so well as previous bag-of-word model, perhaps due to the noise in the texts, and the text are quite complicated.

# In[22]:


import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
svd = TruncatedSVD(n_components=10)
svd.fit(X_train_tfidf) 


# In[23]:


#Check the most important words within each topic,namely each component
for i, component in enumerate(svd.components_):
    #The most important 20 words' index
    nlargest = component.argsort()[-10:]
    word_list = []
    for n in nlargest:
        word_list.append(feature_words[n])
    print('*'*20)
    print('The ' + str(i) + 'th component:')
    print(word_list)


# In[32]:


#Specify the number of topics
n_components = [5, 10, 15, 20, 25,30]
#Traverse each 
plt.figure(figsize=(10, 12))
for i, model in enumerate(models):
    train_scores = []
    test_scores = []
    print('Start to run', str(model.__class__))
    for n_component in n_components:
        svd = TruncatedSVD(n_components=n_component)
        lda = svd.fit(X_train_tfidf) 
       #Make pipeline to simplify code
    
        pipeline = make_pipeline(lda, model)
        pipeline.fit(X_train_tfidf, newsgroups_train.Type)
        train_pred = pipeline.predict(X_train_tfidf)
        test_pred = pipeline.predict(X_test_tfidf)
        test_scores.append(f1_score(newsgroups_test.Type, test_pred, average='micro'))
        train_scores.append(f1_score(newsgroups_train.Type, train_pred, average='micro'))
    plt.subplot(3, 2 , i+1)
    title = str(model.__class__).replace("'>", '').split('.')[-1]
    make_plot(train_scores, test_scores, title) 
    plt.axis((0, 6, 0, 1))
    plt.tight_layout() 


# In[36]:




