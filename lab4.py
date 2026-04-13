from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from nltk.classify import MaxentClassifier
texts =["good movie","bad movie", "excellent audio","worst audio"]
labels=["pos","neg","pos","neg"]
cv = CountVectorizer()
X= cv.fit_transform(texts)
model = MultinomialNB()
model.fit(X,labels)
text = ["excellent movie"]
input = cv.transform(text)
print(model.predict(input))

input=["i love playing cricket","cricket is sport",
       "coding is fun", "coding is imp"]
X= cv.fit_transform(input)
model = KMeans(n_clusters=2)
model.fit(X)
print(model.labels_)
train_data =[ 
        ({"word":"good"},"positive"),
        ({"word":"bad"},"negitive")
        ]
model = MaxentClassifier.train(train_data,max_iter=10)

model.classify(({"word":"good"}))