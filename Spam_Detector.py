#!/usr/bin/env python
# coding: utf-8

# ### SPAM Ham Detection

# In[1]:


import random
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


# In[2]:


## Reading the given dataset
spam = pd.read_csv("C:/Users/Surbhi Ambastha/Desktop/Flask_ML/SMSSpamCollection.txt", sep = "\t", names=["label", "message"])


# In[3]:


print(spam.head())


# In[5]:


## Converting the read dataset in to a list of tuples, each tuple(row) contianing the message and it's label
data_set = []
for index,row in spam.iterrows():
    data_set.append((row['message'], row['label']))


# In[6]:


print(data_set[:5])


# In[7]:


print(len(data_set))


# ### Preprocessing

# In[8]:


## initialise the inbuilt Stemmer and the Lemmatizer
stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()


# In[9]:


def preprocess(document, stem=True):
    'changes document to lower case, removes stopwords and lemmatizes/stems the remainder of the sentence'

    # change sentence to lower case
    document = document.lower()

    # tokenize into words
    words = word_tokenize(document)

    # remove stop words
    words = [word for word in words if word not in stopwords.words("english")]

    if stem:
        words = [stemmer.stem(word) for word in words]
    else:
        words = [wordnet_lemmatizer.lemmatize(word, pos='v') for word in words]

    # join words to make sentence
    document = " ".join(words)

    return document


# In[10]:


## - Performing the preprocessing steps on all messages
messages_set = []
for (message, label) in data_set:
    words_filtered = [e.lower() for e in preprocess(message, stem=False).split() if len(e) >= 3]
    messages_set.append((words_filtered, label))


# In[11]:


print(messages_set[:5])


# ### Preparing to create features

# In[12]:


## - creating a single list of all words in the entire dataset for feature list creation

def get_words_in_messages(messages):
    all_words = []
    for (message, label) in messages:
      all_words.extend(message)
    return all_words


# In[13]:


## - creating a final feature list using an intuitive FreqDist, to eliminate all the duplicate words
## Note : we can use the Frequency Distribution of the entire dataset to calculate Tf-Idf scores like we did earlier.

def get_word_features(wordlist):

    #print(wordlist[:10])
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


# In[14]:


## - creating the word features for the entire dataset
word_features = get_word_features(get_words_in_messages(messages_set))
print(len(word_features))


# ### Preparing to create a train and test set

# In[15]:


## - creating slicing index at 80% threshold
sliceIndex = int((len(messages_set)*.8))


# In[16]:


## - shuffle the pack to create a random and unbiased split of the dataset
random.shuffle(messages_set)


# In[17]:


train_messages, test_messages = messages_set[:sliceIndex], messages_set[sliceIndex:]


# In[18]:


len(train_messages)
len(test_messages)


# ### Preparing to create feature maps for train and test data

# In[19]:


## creating a LazyMap of feature presence for each of the 8K+ features with respect to each of the SMS messages
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


#cv = extract_features(document)
#import pickle
#pickle.dump(cv, open('tranform.pkl', 'wb'))
# In[20]:


## - creating the feature map of train and test data

training_set = nltk.classify.apply_features(extract_features, train_messages)
testing_set = nltk.classify.apply_features(extract_features, test_messages)


# In[21]:


print(training_set[:5])


# In[22]:


print('Training set size : ', len(training_set))
print('Test set size : ', len(testing_set))


# ### Training

# In[23]:


## Training the classifier with NaiveBayes algorithm
spamClassifier = nltk.NaiveBayesClassifier.train(training_set)


# ### Evaluation

# In[24]:


## - Analyzing the accuracy of the test set
#print(nltk.classify.accuracy(spamClassifier, training_set))


# In[25]:


## Analyzing the accuracy of the test set
#print(nltk.classify.accuracy(spamClassifier, testing_set))


# In[26]:


## Testing a example message with our newly trained classifier



m = 'CONGRATULATIONS!! As a valued account holder you have been selected to receive a £900 prize reward! Valid 12 hours only.'
print('Classification result : ', spamClassifier.classify(extract_features(m.split())))

messageinput = ""

predictnow = spamClassifier.classify(extract_features(messageinput.split()))

print(predictnow)

import pickle
f = open('predict_classifier.pickle', 'wb')
pickle.dump(predictnow,f)
print('Classifier stored at ', f.name)
f.close()


# In[27]:


## Priting the most informative features in the classifier
#print(spamClassifier.show_most_informative_features(50))


# In[28]:


## storing the classifier on disk for later usage
import pickle
f = open('spam_classifier.pickle', 'wb')
pickle.dump(spamClassifier,f)
print('Classifier stored at ', f.name)
f.close()

