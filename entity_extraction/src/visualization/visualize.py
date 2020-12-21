import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
nltk.download()
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns

data=pd.read_csv(r'C:\Users\Lovely\PycharmProjects\Entity_Extraction_Invoice\entity_extraction\src\data\Required_Dataset3.csv',encoding='utf-8')
print(data.head())

# Creating word cloud , to see what type of InvoiceNumber isin our data ..similar way we can do for each entity to better understand our data
text=''
data1 = data[data['Label'] == 'B-InvoiceNumber']
for entity in data1['Data']:
    text += entity
wordcloud = WordCloud(
    width=300,
    height=300,
    background_color='black',
    stopwords=set(nltk.corpus.stopwords.words("english"))).generate(str(text))
fig = plt.figure(
    figsize=(5, 5),
    facecolor='k',
    edgecolor='k')
plt.imshow(wordcloud, interpolation='bilinear')
plt.show()

# #  Below code for N Gram anaylysis to check distribution of word in data , unigram and bigram data visualization


def unigrams(input_data, n=None):

    vector = CountVectorizer(ngram_range=(1, 1)).fit(input_data)
    bag_of_words = vector.transform(input_data)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vector.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def bigrams(input_data, n=None):

    vector = CountVectorizer(ngram_range=(2, 2)).fit(input_data)
    bag_of_words = vector.transform(input_data)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vector.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

fig, axes = plt.subplots(ncols=2, figsize=(20,20), dpi=50)

top_unigrams=unigrams(data['Data'][:])
x,y=map(list,zip(*top_unigrams))

sns.barplot(x=y,y=x, ax=axes[0], color='teal')


top_bigrams=bigrams(data['Data'][:])
x,y=map(list,zip(*top_bigrams))

sns.barplot(x=y,y=x, ax=axes[1], color='crimson')

axes[0].set_title(' unigrams (single pair) in data', fontsize=20)
axes[1].set_title('bigrams(2 together) in data', fontsize=20)

plt.show()


# # From above we can see  unigram and bigram in our data
