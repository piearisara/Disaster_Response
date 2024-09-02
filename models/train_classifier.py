import sys
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath: str):
    """
    Get features, and label
    """
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table("DisasterResponse", 
                           engine)
    X = df['message']
    Y = df.iloc[:, 4:] # hard code TODO: FIX
    category_names = Y.columns.tolist()
    return X, Y, category_names


def tokenize(text: str):
    """
    tokenize the string
    """
    #regex
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
   
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words("english")]
    return tokens


def build_model():
    """
    Build machine learning model
    """
    #svd to reduce dim for faster training
    #src: https://scikit-learn.org/0.19/auto_examples/model_selection/grid_search_text_feature_extraction.html
    #https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('svd', TruncatedSVD(n_components=100)),  # Reduce dimensionality
        ('clf', MultiOutputClassifier(RandomForestClassifier(
                            n_estimators=2))) 
    ])
    #params for grid search
    parameters = {
        'clf__estimator__n_estimators': [4],
        'clf__estimator__min_samples_split': [2, 4],
    }
    #search
    model = GridSearchCV(pipeline, 
                         param_grid=parameters, 
                         verbose=2)

    return model


def evaluate_model(model, X_test, Y_test, category_names) -> None:
    Y_pred = model.predict(X_test)
    #src: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    for i, column in enumerate(category_names):
        print(f"Cat: {column}:", classification_report(Y_test.iloc[:, i], Y_pred[:, i]))


def save_model(model, model_filepath: str) -> None:
    """
    save the model checkpoint
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
      
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()