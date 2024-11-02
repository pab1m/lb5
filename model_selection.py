import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


def model_training(df):
    X = df.drop(columns='Status')
    y = df['Status']

    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

    model = LogisticRegression(class_weight='balanced', C=1, max_iter=500, penalty='l1', solver='liblinear')

    print('Training LogisticRegression...')
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    print('Test set metrics:\n', metrics.classification_report(y_test, y_pred_test))

    y_pred_valid = model.predict(X_valid)
    print('Validation set metrics:\n', metrics.classification_report(y_valid, y_pred_valid))

    joblib.dump(model, 'model.pkl')
