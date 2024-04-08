import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
import nltk

# Read and preprocess the training data
#Stripping every variables into each individual vairables then sorting them back into a (token, pos_tag) format
def read_conll_data(filename):
    sentences = []
    sentence = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line == '':
                if sentence:
                    sentences.append(sentence)
                sentence = []
            else:
                token, pos_tag, _ = line.split()
                sentence.append((token, pos_tag))
    return sentences

# Feature extraction
# Sentence will be a list of (token, pos_tag) tuples and index will be the position of the token in the sentence
def extract_features(sentence, index):
    features = {} # This is empty dictionary, we will add features to it
    token, _ = sentence[index]

    # Basic features
    features['token'] = token.lower() # Store the lowerase verion of the token

    #Check the token is given index is the first or last token in the sentence
    #If yes, then store the String <START> to prev_token
    features['prev_token'] = sentence[index - 1][0].lower() if index > 0 else '<START>'

    #Check the token is not in the last position of the sentence it is last token then store the String <END> to next_token
    features['next_token'] = sentence[index + 1][0].lower() if index < len(sentence) - 1 else '<END>'
    
    # Enhanced features
    
    #Check the token is all uppercase save the boolean value to is_upper
    features['is_upper'] = token.isupper()

    #Check the token is title case save the boolean value to is_title    
    features['is_title'] = token.istitle()

    #Check the token is a digit save the boolean value to is_digit
    features['is_digit'] = token.isdigit()

    #Check the token is a punctuation save suffix
    features['suffix'] = token[-3:]

    #Check the token is a punctuationn save prefix
    features['prefix'] = token[:3]

    #Check the token's length
    features['length'] = len(token)

    #Extract the last two characters of the token. This is shorter version of the token
    features['token_2gram'] = token[-2:]

    #Extract the last three characters of the token. This is shorter version of the token
    features['token_3gram'] = token[-3:]

    #Check the token if a '-' is presented in the token
    features['has_hyphen'] = '-' in token

    #Check the token is alphanumeric 
    features['is_alphanumeric'] = token.isalnum()

    return features

# Prepare training data and labels
def prepare_data(sentences):

    #X stores feature dictionaries any Y stores labels
    X = []

    #Y sotres the corresponding part of speech tags
    y = []

    #Iterate over sentences and tokens and extract features
    for sentence in sentences:

        #Iterate over tokens in the sentence
        for i in range(len(sentence)):
            X.append(extract_features(sentence, i))
            y.append(sentence[i][1])
    return X, y

if __name__ == "__main__":
    # Read the training data
    train_data = read_conll_data('train.txt')

    # Split data into training and development sets
    # train_test_split will split the data into 80% training and 20% development data. Utility function for sklearn
    # 0.2 data will be used for development and 0.8 data will be used for training
    # random_state is used to ensure that the same split is generated every time the code is run
    train_sentences, dev_sentences = train_test_split(train_data, test_size=0.2, random_state=42)

    # Prepare training and development data
    X_train, y_train = prepare_data(train_sentences)
    X_dev, y_dev = prepare_data(dev_sentences)

    # Vectorize features

    # Use DictVectorizer to convert feature dictionaries into feature vectors
    # Each token's feature were dictionaries it need to convert into feature vectors
    vectorizer = DictVectorizer()

    #This learns feature names from traning data and assigns an index to each feature in X_train
    X_train = vectorizer.fit_transform(X_train)

    #Convert the feature dictionaries in the feature dictionary becomes a column in the matrix
    #If key exist the value will placed in the column otherwise 0 will be placed in the column
    #If will result in a spase matrix where each row corresponds to a token and each column corresponds to a feature 
    X_dev = vectorizer.transform(X_dev)

    # Hyperparameter tuning for Logistic Regression

    # This line defines a dictionary called parameters. 
    # It contains a set of hyperparameters and their corresponding values that will be used for tuning the Logistic Regression model.
    parameters = {'C': [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10], 'max_iter': [1000], 'solver': ['lbfgs', 'sag', 'saga']}

    #GridSearchCV is a method for hyperparameter tuning that exhaustively searches over a specified hyperparameter grid.
    #LogisticRegression(): This is the base model that we want to tune. It creates an instance of the Logistic Regression classifier.
    #cv=5: This specifies a 5-fold cross-validation, which means the dataset is split into 5 parts, and the model is trained and validated five times.
    logistic_regression = GridSearchCV(LogisticRegression(), parameters, cv=5, scoring='f1_weighted')

    #This line fits (trains) the GridSearchCV object on the training data (X_train and y_train). 
    logistic_regression.fit(X_train, y_train)

    # Evaluate the best Logistic Regression model
    #After fitting the grid search, this line retrieves the best model (Logistic Regression classifier) that was found during the hyperparameter tuning process.
    best_model = logistic_regression.best_estimator_

    #This line uses the best model to make predictions on the development/validation dataset X_dev, storing the predicted labels in y_pred.
    y_pred = best_model.predict(X_dev)

    #This line calculates the accuracy of the model's predictions by comparing them to the true labels in y_dev.
    accuracy = accuracy_score(y_dev, y_pred)

    #This line calculates the weighted F1-score of the model's predictions. The 'weighted' average is used because it accounts for class imbalance.
    f1 = f1_score(y_dev, y_pred, average='weighted')
    
    print(f'Logistic Regression Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}')
