import locale
import glob
import os.path
import requests
import tarfile
import sys
import codecs
import smart_open

dirname = 'aclImdb'
filename = 'aclImdb_v1.tar.gz'
locale.setlocale(locale.LC_ALL, 'C')

if sys.version > '3':
    control_chars = [chr(0x85)]
else:
    control_chars = [unichr(0x85)]

# Convert text to lower-case and strip punctuation/symbols from words
def normalize_text(text):
    norm_text = text.lower()
    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')
    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')
    return norm_text

import time
start = time.clock()

if not os.path.isfile('aclImdb/alldata-id.txt'):
    if not os.path.isdir(dirname):
        if not os.path.isfile(filename):
            # Download IMDB archive
            print("Downloading IMDB archive...")
            url = u'http://ai.stanford.edu/~amaas/data/sentiment/' + filename
            r = requests.get(url)
            with open(filename, 'wb') as f:
                f.write(r.content)
        tar = tarfile.open(filename, mode='r')
        tar.extractall()
        tar.close()

    # Concatenate and normalize test/train data
    print("Cleaning up dataset...")
    folders = ['train/pos', 'train/neg', 'test/pos', 'test/neg', 'train/unsup']
    alldata = u''
    for fol in folders:
        temp = u''
        output = fol.replace('/', '-') + '.txt'
        # Is there a better pattern to use?
        txt_files = glob.glob(os.path.join(dirname, fol, '*.txt'))
        for txt in txt_files:
            with smart_open.smart_open(txt, "rb") as t:
                t_clean = t.read().decode("utf-8")
                for c in control_chars:
                    t_clean = t_clean.replace(c, ' ')
                temp += t_clean
            temp += "\n"
        temp_norm = normalize_text(temp)
        with smart_open.smart_open(os.path.join(dirname, output), "wb") as n:
            n.write(temp_norm.encode("utf-8"))
        alldata += temp_norm

    with smart_open.smart_open(os.path.join(dirname, 'alldata-id.txt'), 'wb') as f:
        for idx, line in enumerate(alldata.splitlines()):
            num_line = u"_*{0} {1}\n".format(idx, line)
            f.write(num_line.encode("utf-8"))

end = time.clock()
print ("Total running time: ", end-start)





import os.path
assert os.path.isfile("aclImdb/alldata-id.txt"), "alldata-id.txt unavailable"



import gensim
from gensim.models.doc2vec import TaggedDocument
from collections import namedtuple

SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')

alldocs = []  # Will hold all docs in original order
with open('aclImdb/alldata-id.txt', encoding='utf-8') as alldata:
    for line_no, line in enumerate(alldata):
        tokens = gensim.utils.to_unicode(line).split()
        words = tokens[1:]
        tags = [line_no] # 'tags = [tokens[0]]' would also work at extra memory cost
        split = ['train', 'test', 'extra', 'extra'][line_no//25000]  # 25k train, 25k test, 25k extra
        sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][line_no//12500] # [12.5K pos, 12.5K neg]*2 then unknown
        alldocs.append(SentimentDocument(words, tags, split, sentiment))

train_docs = [doc for doc in alldocs if doc.split == 'train']
test_docs = [doc for doc in alldocs if doc.split == 'test']
doc_list = alldocs[:]  # For reshuffling per pass

print('%d docs: %d train-sentiment, %d test-sentiment' % (len(doc_list), len(train_docs), len(test_docs)))


# ## Set-up Doc2Vec Training & Evaluation Models



from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import OrderedDict
import multiprocessing

cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

simple_models = [
    # PV-DM w/ concatenation - window=5 (both sides) approximates paper's 10-word total window size
    Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=cores),
    # PV-DBOW 
    Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=2, workers=cores),
    # PV-DM w/ average
    Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=5, hs=0, min_count=2, workers=cores),
]

# Speed up setup by sharing results of the 1st model's vocabulary scan
simple_models[0].build_vocab(alldocs)  # PV-DM w/ concat requires one special NULL word so it serves as template
print(simple_models[0])
for model in simple_models[1:]:
    model.reset_from(simple_models[0])
    print(model)

models_by_name = OrderedDict((str(model), model) for model in simple_models)



from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import OrderedDict
import multiprocessing

cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

simple_models = [
    # PV-DM w/ concatenation - window=5 (both sides) approximates paper's 10-word total window size
    Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=cores),
    # PV-DBOW 
    Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=2, workers=cores),
    # PV-DM w/ average
    Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=5, hs=0, min_count=2, workers=cores),
]

# Speed up setup by sharing results of the 1st model's vocabulary scan
simple_models[0].build_vocab(alldocs)  # PV-DM w/ concat requires one special NULL word so it serves as template
print(simple_models[0])
for model in simple_models[1:]:
    model.reset_from(simple_models[0])
    print(model)

models_by_name = OrderedDict((str(model), model) for model in simple_models)

from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])


# ## Predictive Evaluation Methods



import numpy as np
import statsmodels.api as sm
from random import sample
from sklearn.svm import SVC
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# For timing
from contextlib import contextmanager
from timeit import default_timer
import time 

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

    
    
def logistic_predictor_from_data(train_targets, train_regressors):
    clf = SVC(decision_function_shape='ovr', kernel=my_kernel)
    predictor = clf.fit(train_regressors,train_targets)
    return predictor

def error_rate_for_model(test_model, train_set, test_set, infer=False, infer_steps=3, infer_alpha=0.1, infer_subsample=0.1):
    """Report error rate on test_doc sentiments, using supplied model and train_docs"""


    train_targets, train_regressors = zip(*[(doc.sentiment, test_model.docvecs[doc.tags[0]]) for doc in train_set])
    train_regressors = sm.add_constant(train_regressors)
    predictor = logistic_predictor_from_data(train_targets, train_regressors)

    test_data = test_set
    if infer:
        if infer_subsample < 1.0:
            test_data = sample(test_data, int(infer_subsample * len(test_data)))
        test_regressors = [test_model.infer_vector(doc.words, steps=infer_steps, alpha=infer_alpha) for doc in test_data]
    else:
        test_regressors = [test_model.docvecs[doc.tags[0]] for doc in test_docs]
    test_regressors = sm.add_constant(test_regressors)
    
    # Predict & evaluate
    test_predictions = predictor.predict(test_regressors)
    corrects = sum(np.rint(test_predictions) == [doc.sentiment for doc in test_data])
    errors = len(test_predictions) - corrects
    error_rate = float(errors) / len(test_predictions)
    return (error_rate, errors, len(test_predictions), predictor)


# ## Bulk Training


from collections import defaultdict
best_error = defaultdict(lambda: 1.0)  # To selectively print only best errors achieved



from random import shuffle
import datetime

alpha, min_alpha, passes = (0.025, 0.001, 5)
alpha_delta = (alpha - min_alpha) / passes

print("START %s" % datetime.datetime.now())

for kernel in ['rbf']:
    print('kernel is', kernel)
    my_kernel = kernel




    from gensim.models import Doc2Vec
    import gensim.models.doc2vec
    from collections import OrderedDict
    import multiprocessing

    cores = multiprocessing.cpu_count()
    assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

    simple_models = [
        # PV-DM w/ concatenation - window=5 (both sides) approximates paper's 10-word total window size
        Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=cores),
        # PV-DBOW 
        Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=2, workers=cores),
        # PV-DM w/ average
        Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=5, hs=0, min_count=2, workers=cores),
    ]

    # Speed up setup by sharing results of the 1st model's vocabulary scan
    simple_models[0].build_vocab(alldocs)  # PV-DM w/ concat requires one special NULL word so it serves as template
    print(simple_models[0])
    for model in simple_models[1:]:
        model.reset_from(simple_models[0])
        print(model)

    models_by_name = OrderedDict((str(model), model) for model in simple_models)

    from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
    models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
    models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])


    for epoch in range(passes):
        shuffle(doc_list)  # Shuffling gets best results

        for name, train_model in models_by_name.items():
            # Train
            duration = 'na'
            train_model.alpha, train_model.min_alpha = alpha, alpha
            with elapsed_timer() as elapsed:
                train_model.train(doc_list, total_examples=len(doc_list), epochs=1)
                duration = '%.1f' % elapsed()

            # Evaluate
            eval_duration = ''
            with elapsed_timer() as eval_elapsed:
                err, err_count, test_count, predictor = error_rate_for_model(train_model, train_docs, test_docs)
            eval_duration = '%.1f' % eval_elapsed()
            best_indicator = ' '
            if err <= best_error[name]:
                best_error[name] = err
                best_indicator = '*' 
            print("%s%f : %i passes : %s %ss %ss" % (best_indicator, err, epoch + 1, name, duration, eval_duration))

            
            if ((epoch + 1) % 5) == 0 or epoch == 0:
                eval_duration = ''
                with elapsed_timer() as eval_elapsed:
                    infer_err, err_count, test_count, predictor = error_rate_for_model(train_model, train_docs, test_docs, infer=True)
                eval_duration = '%.1f' % eval_elapsed()
                best_indicator = ' '
                if infer_err < best_error[name + '_inferred']:
                    best_error[name + '_inferred'] = infer_err
                    best_indicator = '*'
                print("%s%f : %i passes : %s %ss %ss" % (best_indicator, infer_err, epoch + 1, name + '_inferred', duration, eval_duration))

        print('Completed pass %i at alpha %f' % (epoch + 1, alpha))
        alpha -= alpha_delta

    print("END %s" % str(datetime.datetime.now()))
    
    # Print best error rates achieved
    print("Err rate Model")
    for rate, name in sorted((rate, name) for name, rate in best_error.items()):
        print("%f %s" % (rate, name))
    


# In[10]:


import pickle

with open('model.pickle4', 'wb') as handle:
    pickle.dump(models_by_name, handle, protocol=pickle.HIGHEST_PROTOCOL)




# ### Are inferred vectors close to the precalculated ones?


doc_id = np.random.randint(simple_models[0].docvecs.count)  # Pick random doc; re-run cell for more examples
print('for doc %d...' % doc_id)
for model in simple_models:
    inferred_docvec = model.infer_vector(alldocs[doc_id].words)
    print('%s:\n %s' % (model, model.docvecs.most_similar([inferred_docvec], topn=3)))


# ### Do close documents seem more related than distant ones?



import random

doc_id = np.random.randint(simple_models[0].docvecs.count)  # pick random doc, re-run cell for more examples
model = random.choice(simple_models)  # and a random model
sims = model.docvecs.most_similar(doc_id, topn=model.docvecs.count)  # get *all* similar documents
print(u'TARGET (%d): «%s»\n' % (doc_id, ' '.join(alldocs[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(alldocs[sims[index][0]].words)))



# ### Do the word vectors show useful similarities?

# In[14]:


word_models = simple_models[:]


# In[15]:


import random
from IPython.display import HTML
# pick a random word with a suitable number of occurences
while True:
    word = random.choice(word_models[0].wv.index2word)
    if word_models[0].wv.vocab[word].count > 10:
        break
# or uncomment below line, to just pick a word from the relevant domain:
#word = 'comedy/drama'
similars_per_model = [str(model.most_similar(word, topn=20)).replace('), ','),<br>\n') for model in word_models]
similar_table = ("<table><tr><th>" +
    "</th><th>".join([str(model) for model in word_models]) + 
    "</th></tr><tr><td>" +
    "</td><td>".join(similars_per_model) +
    "</td></tr></table>")
print("most similar words for '%s' (%d occurences)" % (word, simple_models[0].wv.vocab[word].count))
HTML(similar_table)


# In[16]:


import random
from IPython.display import HTML
# pick a random word with a suitable number of occurences
while True:
    word = random.choice(word_models[0].wv.index2word)
    if word_models[0].wv.vocab[word].count > 30:
        break
# or uncomment below line, to just pick a word from the relevant domain:
#word = 'comedy/drama'
similars_per_model = [str(model.most_similar(word, topn=20)).replace('), ','),<br>\n') for model in word_models]
similar_table = ("<table><tr><th>" +
    "</th><th>".join([str(model) for model in word_models]) + 
    "</th></tr><tr><td>" +
    "</td><td>".join(similars_per_model) +
    "</td></tr></table>")
print("most similar words for '%s' (%d occurences)" % (word, simple_models[0].wv.vocab[word].count))
HTML(similar_table)


# Do the DBOW words look meaningless? That's because the gensim DBOW model doesn't train word vectors – they remain at their random initialized values – unless you ask with the `dbow_words=1` initialization parameter. Concurrent word-training slows DBOW mode significantly, and offers little improvement (and sometimes a little worsening) of the error rate on this IMDB sentiment-prediction task. 

if os.path.isfile('questions-words.txt'):
    for model in word_models:
        sections = model.accuracy('questions-words.txt')
        correct, incorrect = len(sections[-1]['correct']), len(sections[-1]['incorrect'])
        print('%s: %0.2f%% correct (%d of %d)' % (model, float(correct*100)/(correct+incorrect), correct, correct+incorrect))



from sklearn.metrics import roc_curve, auc
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# To mix the Google dataset (if locally available) into the word tests...

# In[28]:


from sklearn.preprocessing import scale
def error_rate_for_model_NEW(test_model, train_set, test_set, infer=False, infer_steps=3, infer_alpha=0.1, infer_subsample=0.1):
    """Report error rate on test_doc sentiments, using supplied model and train_docs"""

    print(test_model)
    train_targets, train_regressors = zip(*[(doc.sentiment, test_model.docvecs[doc.tags[0]]) for doc in train_set])
    train_regressors = sm.add_constant(train_regressors)
    predictor = logistic_predictor_from_data(train_targets, train_regressors)

    test_data = test_set
    if infer:
        if infer_subsample < 1.0:
            test_data = sample(test_data, int(infer_subsample * len(test_data)))
        test_regressors = [test_model.infer_vector(doc.words, steps=infer_steps, alpha=infer_alpha) for doc in test_data]
    else:
        test_regressors = [test_model.docvecs[doc.tags[0]] for doc in test_docs]
    test_regressors = sm.add_constant(test_regressors)
    
    # Predict & evaluate
    test_predictions = predictor.predict(test_regressors)
    corrects = sum(np.rint(test_predictions) == [doc.sentiment for doc in test_data])
    errors = len(test_predictions) - corrects
    error_rate = float(errors) / len(test_predictions)
    
    correct_vector = []
    #Create ROC curve
    for doc in test_data:
        correct_vector.append(doc.sentiment)

    fpr,tpr,_ = roc_curve(correct_vector, test_predictions)
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')

    plt.show()
    
    return (error_rate, errors, len(test_predictions), predictor)



#Test on test set -  check overfitting


from random import shuffle
import datetime
from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import OrderedDict
import multiprocessing
#Create ROC curve
from sklearn.metrics import roc_curve, auc
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


print("START %s" % datetime.datetime.now())

for kernel in ['rbf']:
    print('kernel is', kernel)
    my_kernel = kernel

    cores = multiprocessing.cpu_count()
    assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"


    for name, train_model in models_by_name.items():
        # Evaluate
        eval_duration = ''
        with elapsed_timer() as eval_elapsed:
            err, err_count, test_count, predictor = error_rate_for_model_NEW(train_model, train_docs, test_docs)
        eval_duration = '%.1f' % eval_elapsed()
        best_indicator = ' '
        if err <= best_error[name]:
            best_error[name] = err
            best_indicator = '*' 
        print(err,  name, duration)
        





