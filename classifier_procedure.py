# Arda Mavi
from sklearn.externals import joblib

def getClassifiers():
    # Getting trained classifier:
    try:
        rClf = joblib.load('Classifier/R_Classifier.pkl')
        gClf = joblib.load('Classifier/G_Classifier.pkl')
        bClf = joblib.load('Classifier/B_Classifier.pkl')
    except:
        import os
        if not os.path.exists('Classifier'):
            os.makedirs('Classifier')
        return None
    return [rClf, gClf, bClf]

def trainClassifier(clf, X, y):
    # Training classifier:
    return clf.fit(X, y)

def getScore(clf, X, y):
    # Get score:
    return clf.score(X, y)

def getPredict(clfs, img):
    # Get predict:
    predicts = []
    for clf in clfs:
        predicts.append(clf.predict(img))
    return predicts

def saveClassifier(clf, dir):
    # Save classifier:
    joblib.dump(clf, dir)
