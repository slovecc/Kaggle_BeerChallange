def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    '''errors = np.abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    '''
    accuracy=metrics.accuracy_score(predictions, test_labels)
    print('Accuracy = ',(accuracy))
    return accuracy

def xg_boost(X_train, X_test, y_train, y_test):
    import pandas as pd
    import numpy as np

    from sklearn.model_selection import RandomizedSearchCV  # Number of trees in random forest
    from sklearn.feature_selection import SelectFromModel
    from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GridSearchCV

    import xgboost as xgb
    ###### random search of optimum parameters for the model #############

    ''' n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(start=3, stop=20, num=5)]

    learning_rate = [float(x) for x in np.linspace(start=0.01, stop=0.2, num=10)]

    subsample = [float(x) for x in np.linspace(start=0.5, stop=1.2, num=4)]

    colsample_bytree = [float(x) for x in np.linspace(start=0.4, stop=1.2, num=4)]

    gamma = [int(x) for x in np.linspace(start=1, stop=8, num=5)]

    random_grid = {'n_estimators': n_estimators,
                   'learning_rate': learning_rate,
                   'max_depth': max_depth,
                   'subsample': subsample,
                   'colsample_bytree': colsample_bytree,
                   'gamma': gamma}
    rf = xgb.XGBClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                   n_iter=10, cv=4, verbose=2, random_state=42,
                                   n_jobs=-1)  # Fit the random search model
    rf_random.fit(X_train, y_train)

    best_random=rf_random.best_estimator_


    random_accuracy = evaluate(best_random, X_test, y_test) '''
    model_xgb = xgb.XGBClassifier(n_estimators=300,
                              learning_rate=0.095,
                              max_depth=7,
                              subsample=0.73,
                              colsample_bytree=0.67,
                              gamma=6,
                              colsample_bynode=1)

    ###### cross validation to check the accuracy of the parameter of best model #############
    '''kfold = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)

    results = cross_val_score(rf, X_train, y_train, cv=kfold)
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

    model_xgb=model_xgb.fit(X_train, y_train)

    evaluate(model_xgb, X_test, y_test)'''
    ##unisci test and train
    X_all = X_train.append(X_test)
    y_all = y_train.append(y_test)
    model_xgb = model_xgb.fit(X_all, y_all)

    ##INTERESTING: it gives the probability of the class estimated!!
    bb = model_xgb.predict_proba(X_test)
    bb = pd.DataFrame(bb)

    max_b = bb.max(axis=1)
    max_column = bb.idxmax(axis=1)
    final_max_xg = pd.concat([max_b, max_column], axis=1)

    return (model_xgb, final_max_xg)