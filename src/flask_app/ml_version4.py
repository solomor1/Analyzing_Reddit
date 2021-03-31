from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from io import StringIO
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import seaborn as sns
from IPython.display import display
from sklearn import metrics

# https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
# getting data
conn = sqlite3.connect("./reddit_comments.db")
c = conn.cursor()

c.execute('''SELECT commentBody, Subreddit, commentScore, Sentiment FROM allData WHERE LENGTH(commentBody) < 100 and LENGTH(commentBody) > 10 and Subreddit != "dankmemes" and Subreddit != "nba" and Subreddit != "soccer" ORDER BY Random() LIMIT 50000; ''')
data = c.fetchall()

data_df = pd.DataFrame(data)
data_df.columns = ['comment', 'subreddit', 'score', 'sentiment']


data_df['category_id'] = data_df['subreddit'].factorize()[0]
category_id_df = data_df[['subreddit', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id','subreddit']].values)
#print data_df[:]

fig = plt.figure(figsize=(8,6))
# map number of comments to subreddit
data_df.groupby('subreddit').comment.count().plot.bar(ylim=0)
#plt.show()
#imblanced - most commnts are in /r/soccor


 # caluclate tfidf vector for each comment

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1,2), stop_words='english')
features = tfidf.fit_transform(data_df.comment).toarray()
labels = data_df.category_id
#print features.shape

# find the terms that are the most correlated with each subreddit
N = 2
for subreddit, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [ v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    #print("# '{}':".format(subreddit))
    #print(" . Most correlated unigrams:\n. {}".format('\n.'.join(unigrams[-N:])))
    #print(" . most correlated bigrams:\n. {}".format('\n.'.join(bigrams[-N:])))

# time to train the classifiers!
'''X_train, X_test, y_train, y_test = train_test_split(data_df['comment'], data_df['subreddit'], random_state=0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#clf = MultinomialNB().fit(X_train_tfidf, y_train)


models = [ RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
           LinearSVC(),
           MultinomialNB(),
           LogisticRegression(random_state=0),]

CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx','accuracy'])'''

#sns.boxplot(x='model_name', y='accuracy', data=cv_df)
#sns.stripplot(x='model_name', y='accuracy', data=cv_df, size=8, jitter=True, edgecolor="gray", linewidth=2)
#plt.show()

#print cv_df.groupby('model_name').accuracy.mean()

# linear svc has highest accuracy 38%
#model = LinearSVC()
model = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, data_df.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=category_id_df.subreddit.values, yticklabels=category_id_df.subreddit.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
# diagonal means that actual = predicted
plt.save('fig.png')


print (metrics.classification_report(y_test, y_pred, target_names=data_df['subreddit']))










# lets have a closer look at the misclassifications cause
'''for predicted in category_id_df.category_id:
    for actual in category_id_df.category_id:
        if predicted != actual and conf_mat[actual, predicted] >= 10:
            print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual],
            id_to_category[predicted], conf_mat[actual, predicted]))
            display(data_df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['subreddit', 'comment']])
            print('')

# There were quite a few misclassifications because a lot of the comments have quite vague language

model.fit(features, labels)

N=2

for subreddit, category_id in sorted(category_to_id.items()):
    indices = np.argsort(model.coef_[category_id])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
    bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
    print("# '{}':".format(subreddit))
    print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams).encode('utf-8')))
    print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams).encode('utf-8')))

print " "'''
