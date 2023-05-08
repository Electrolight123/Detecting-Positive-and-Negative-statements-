from sklearn import tree 
from sklearn.feature_extraction.text import CountVectorizer
positive_set = [
'we are happy',
'we are the best',
'we will win',
'we are lucky',
'we are loved']

#print(positive_set)

negative_set=['we are sad','we are the worst','we will lose','we are unlucky','we are hated','I am ugly']

#print(negative_set)

sample_set=['we are happy','we are the best','we are sad','we are the worst','they will win']

print(sample_set)

data_set=positive_set+negative_set
# print(data_set)


data_labels=["POSITIVE"] * len(positive_set) + ["NEGATIVE"] * len(negative_set)

#print(data_labels)
'''
from sklearn.feature_extraction.text import CountVectorizer
sample=['cat dog mouse bird','dog bird cow horse cow','cat dog bird fish']
print(sample)

vectorizer = CountVectorizer()
vectorizer.fit(sample)
sample_vectors = vectorizer.transform(sample)
feature_name = vectorizer.get_feature_names_out()
print(feature_name)
print(sample_vectors.toarray())
'''
vectorizer = CountVectorizer()
vectorizer.fit(data_set)
sample_vectors=vectorizer.transform(sample_set)
data_vectors=vectorizer.transform(data_set)
feature_name=vectorizer.get_feature_names_out()
# print(feature_name)
# print(sample_vector.toarray())
# print('  ')
# print(data_vectors.toarray())

classifier  = tree.DecisionTreeClassifier ()
classifier.fit(data_vectors, data_labels)
predictions = classifier.predict (sample_vectors)
print (predictions)