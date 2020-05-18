from flask import Flask, render_template,url_for,request
import random
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


spam = pd.read_csv("SMSSpamCollection.txt", sep = "\t", names=["label", "message"])

data_set = []
for index,row in spam.iterrows():
    data_set.append((row['message'], row['label']))
## initialise the inbuilt Stemmer and the Lemmatizer
stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

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


## - Performing the preprocessing steps on all messages
messages_set = []
for (message, label) in data_set:
    words_filtered = [e.lower() for e in preprocess(message, stem=False).split() if len(e) >= 3]
    messages_set.append((words_filtered, label))

## - creating a single list of all words in the entire dataset for feature list creation

def get_words_in_messages(messages):
    all_words = []
    for (message, label) in messages:
      all_words.extend(message)
    return all_words


## - creating a final feature list using an intuitive FreqDist, to eliminate all the duplicate words

def get_word_features(wordlist):

    #print(wordlist[:10])
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

## - creating the word features for the entire dataset
word_features = get_word_features(get_words_in_messages(messages_set))
print(len(word_features))


# ### Preparing to create a train and test set

## - creating slicing index at 80% threshold
sliceIndex = int((len(messages_set)*.9))


## - shuffle the pack to create a random and unbiased split of the dataset
random.shuffle(messages_set)

train_messages, test_messages = messages_set[:sliceIndex], messages_set[sliceIndex:]
# ### Preparing to create feature maps for train and test data
## creating a LazyMap of feature presence for each of the 8K+ features with respect to each of the SMS messages
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

## - creating the feature map of train and test data

training_set = nltk.classify.apply_features(extract_features, train_messages)

spamClassifier = nltk.NaiveBayesClassifier.train(training_set)


app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
	return render_template('home.html')

@app.route('/recommend')
def recommend():
    message = request.args.get('word')
    data = list(message.split())
    m =' '.join(data)
    r = spamClassifier.classify(extract_features(m.split()))
    if r == 'spam':
        return render_template('recommend.html',x=message,r=r,t='s')
    else:
     return render_template('recommend.html',x=message,r=r,t='r')
        
if __name__ ==  '__main__':
	app.run()
