import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

def spam_detector(train_df, valid_df, test_df):
    # 1. Text Vectorization (TF-IDF with bigrams and min_df=2)
    vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
    
    # Fit and transform on train data, transform on valid and test data
    X_train = vectorizer.fit_transform(train_df['text'])
    X_valid = vectorizer.transform(valid_df['text'])
    X_test = vectorizer.transform(test_df['text'])
    
    y_train = train_df['label']
    y_valid = valid_df['label']
    y_test = test_df['label']
    
    # 2. Create classifiers
    classifiers = {
        'LogisticRegression': LogisticRegression(random_state=0),
        'MultinomialNB': MultinomialNB(),
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=0),
        'LinearSVC': LinearSVC(random_state=0)
    }
    
    best_classifier = None
    best_classifier_name = None
    min_spam_misclassified = float('inf')
    results = {}
    
    # 3. Train and evaluate each classifier
    for clf_name, clf in classifiers.items():
        # Train classifier
        clf.fit(X_train, y_train)
        
        # Make predictions on validation data
        y_valid_pred = clf.predict(X_valid)
        
        # Calculate confusion matrix for validation data
        cm = confusion_matrix(y_valid, y_valid_pred)
        
        # Misclassified spam messages (class 0 predicted as 1)
        spam_misclassified = cm[0, 1]
        
        # Store results
        results[clf_name] = {'confusion_matrix': cm, 'model': clf}
        
        # Determine best classifier based on spam misclassification
        if spam_misclassified < min_spam_misclassified:
            min_spam_misclassified = spam_misclassified
            best_classifier_name = clf_name
            best_classifier = clf
    
    # 4. Tfidf vectorizer for the test data
    tfidf_test_matrix = X_test
    
    # 5. Predict labels for the test data using the best classifier
    y_test_pred = best_classifier.predict(X_test)
    
    # 6. Prepare the final result dictionary
    result = {
        'LogisticRegression': results['LogisticRegression']['confusion_matrix'],
        'MultinomialNB': results['MultinomialNB']['confusion_matrix'],
        'DecisionTreeClassifier': results['DecisionTreeClassifier']['confusion_matrix'],
        'LinearSVC': results['LinearSVC']['confusion_matrix'],
        'BestClassifier': best_classifier_name,
        'TfidfVectorizer': tfidf_test_matrix,
        'Prediction': y_test_pred
    }
    
    return result['BestClassifier']
# Read the CSV files into DataFrames
train_df = pd.read_csv(r'C:\\Kalyani Pakhale\\data-science-Interview\\Machine Learning\\Text_classification\\train.csv')
valid_df = pd.read_csv(r'C:\\Kalyani Pakhale\\data-science-Interview\\Machine Learning\\Text_classification\\valid.csv')
test_df = pd.read_csv(r'C:\\Kalyani Pakhale\\data-science-Interview\\Machine Learning\\Text_classification\\test.csv')

# Call the spam_detector function with the DataFrames
spam_detector(train_df, valid_df, test_df)