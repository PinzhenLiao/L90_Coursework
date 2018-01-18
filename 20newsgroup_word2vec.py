
# coding: utf-8

# 本文中，我们将使用*Python*模块[gensim](http://radimrehurek.com/gensim/models/word2vec.html#id6)对文本训练，生成词向量（即将每个词用向量表示）。使用的数据集为Mikolov所使用的[text8](http://mattmahoney.net/dc/text8.zip)，训练模型为*skip-gram*, *CBOW*。如果你对词向量的概念不甚熟悉，可以拜读下皮果提的[CSDN博客](http://blog.csdn.net/itplus/article/details/37969519)，他深入浅出的介绍了一些列的背景知识、相关概念以及理论推导。

# In[1]:


from collections import Counter
from gensim.models import word2vec 
#from glove import Glove 
#from glove import Corpus 
import argparse
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# 首先，我们读取本地的*text8*数据，关于*gensim*里面的词向量建模工具*word2vec*，可以参看官方文档。

# In[7]:


sentences = word2vec.Text8Corpus('text8')


# ## CBOW模型
# 
# CBOW模型主要思想是给定中心词的上下文环境来推测该中心词，按照官方文档要求，如果我们将*word2vec*函数里面的*sg*参数设置为0，则模型就是*CBOW*（Continuous Bag-Of-Words Model)。这里，我们词向量的长度设置为80，窗户宽度为5，最小频数为3（低于此的词都会被过滤），采用的是*Negative Resampling*方法。

# In[8]:


model = word2vec.Word2Vec(sentences, size=80, sg=1, window=5, min_count=3, workers=4, alpha=0.02, seed=111, negative=6)


# 我们可以查看下词向量，比如"computer"。

# In[9]:


model['computer']


# 我们可以对这些词向量进行可视化，利用PCA降维。为方便起见，我们取词频靠前的500个单词及其词向量，然后用散点图显示。

# In[10]:


#统计词频
word_freq = Counter(model.wv.vocab.keys())
stopwords = ['the', 'of', 'and', 'a', 'to', 's']


# In[11]:


#选取词频靠前的500个单词，除去连接词
freq_words = word_freq.most_common(500)
freq_words = [k for k,v in freq_words if k not in stopwords]


# In[12]:


#匹配词的向量
freq_words_vec = [model[word] for word in freq_words]


# In[13]:


freq_words_vec = np.array(freq_words_vec)


# In[14]:


#将词向量降维为2维向量
model_TSNE = TSNE(n_components=2, random_state=0)
freq_words_vec_2D = model_TSNE.fit_transform(freq_words_vec)


# In[15]:


plt.style.use('ggplot')#设置ggplot风格的背景
def plot(embeddings, labels):
  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(15,15))  # in inches
  for i, label in enumerate(labels):
    x, y = embeddings[i,:]
    #显示散点图并进行标注
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                   ha='right', va='bottom')
  plt.show()
plot(freq_words_vec_2D, freq_words)


# 从词向量散点图可以看出类似词语之间的聚类性，比如"school"、"university"、"college"三个单词距离很近。26个字母也分布在紧邻区域。

# 下面，我们可以利用*gensim*自带的函数之间查看下不同单词之间的相似度。

# In[16]:


model.similarity('boy', 'girl')


# In[17]:


y1 = model.similarity("woman", "man")
print(u"woman和man的相似度为：", y1)
print("--------\n")


# In[18]:


# 计算某个词的相关词列表
y2 = model.most_similar("nice", topn=20)  # 20个最相关的
print(u"和nice最相关的词有：\n")
for item in y2:
    print(item[0], item[1])
print("--------\n")


# In[19]:


# 寻找对应关系
print(' "man" is to "father" as "woman" is to ...? \n')
y3 = model.most_similar(['woman', 'father'], ['man'], topn=3)
for item in y3:
    print(item[0], item[1])
print("--------\n")


# In[20]:


more_examples = ["he his she", "big bigger bad", "going went being"]
for example in more_examples:
    a, b, x = example.split()
    predicted = model.most_similar([x, b], [a])[0][0]
    print("'%s' is to '%s' as '%s' is to '%s'" % (a, b, x, predicted))
print("--------\n")


# In[21]:


# 保存模型，以便重用
model.save("text8.model")
# 对应的加载方式
# model_2 = word2vec.Word2Vec.load("text8.model")


# ## Skip-gram模型
# 
# Skip-gram实际上与CBOW模型很类似，只不过这里是给定中心词来推断上下文环境。在*gensim*里面只需修改参数便可以实现训练过程的变换。

# # Fetch 20Newsgroup Data

# In[22]:


from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train', 
                                  shuffle=True, random_state=11)
newsgroups_test = fetch_20newsgroups(subset='test', 
                                  shuffle=True, random_state=11)


# In[23]:


print("\n".join(newsgroups_train.data[0].split("\n")))


# In[24]:


#数据预处理
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


# ## TiIdf 模型分类

# In[30]:


#利用朴素贝叶斯方法分类
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


# ## 词向量进行分类

# In[33]:


#选择text8词库中的单词
news_words_selected = set(news_words).intersection(set(model.wv.vocab.keys()))


# In[34]:


len(news_words_selected)


# In[35]:


#抽取部分单词的tf-idf向量列
col_indice = [tfidfVectorizer.vocabulary_.get(w) for w in news_words_selected] 
X_train_tfidf_selected = X_train_tfidf[:, col_indice]
X_test_tfidf_selected = X_test_tfidf[:, col_indice]


# In[36]:


#将每个文档所有单词词向量按照权重进行叠，最终生成的向量来表示文档
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


# In[37]:


#将每个文档转化成词向量模式，然后连接成特征矩阵
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


# ## 利用本地20新闻数据训练词向量

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


#将每个文档所有单词词向量按照权重进行叠加
#考虑单词权重
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


#将每个文档转化成词向量模式，然后连接成特征矩阵
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


# 分类的效果不好，很可能样本数量还不够；另外词向量在这里面时静态的，也就是训练完成之后作为静态值输入到分类模型中，可以考虑动态的模型；最后，这里面都是取得向量平均值，没有考虑单词的权重。
