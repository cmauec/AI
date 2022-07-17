"""
https://textblob.readthedocs.io/en/dev/
"""
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob

train = [
    ('I love this sandwich.', 'positive'),
    ('this is an amazing place!', 'positive'),
    ('I feel very good about these beers.', 'positive'),
    ('this is my best work.', 'positive'),
    ("what an awesome view", 'positive'),
    ('I do not like this restaurant', 'negative'),
    ('I am tired of this stuff.', 'negative'),
    ("I can't deal with this", 'negative'),
    ('he is my sworn enemy!', 'negative'),
    ('my boss is horrible.', 'negative')
]

cl = NaiveBayesClassifier(train)
print(cl.classify("This is an amazing library!"))

text = '''
The titular threat of The Blob has always struck me as the ultimate movie
monster: an insatiably hungry, amoeba-like mass able to penetrate
virtually any safeguard, capable of--as a doomed doctor chillingly
describes it--"assimilating flesh on contact.
Snide comparisons to gelatin be damned, it's a concept with the most
devastating of potential consequences, not unlike the grey goo scenario
proposed by technological theorists fearful of
artificial intelligence run rampant.
'''

blob = TextBlob(text)
print(blob.tags, "\n")
print(blob.noun_phrases, "\n")
print(blob.sentences, "\n")

for sentence in blob.sentences:
    print(sentence.sentiment.polarity)
