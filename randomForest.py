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

def random_Forest(X_train, X_test, y_train, y_test):
    import pandas as pd
    import numpy as np

    from sklearn.model_selection import RandomizedSearchCV  # Number of trees in random forest
    from sklearn.feature_selection import SelectFromModel
    from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'log2']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(5, 20, num=2)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 200, 20]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [5, 50, 10]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]  # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=10, cv=4, verbose=2,
                                   random_state=42, n_jobs=-1)  # Fit the random search model
    rf_random.fit(X_train, y_train)

    best_random = rf_random.best_estimator_
    random_accuracy = evaluate(best_random, X_test, y_test)

    rf_random.best_params_

    '''sel = SelectFromModel(RandomForestClassifier(n_estimators=400,
                                                 bootstrap=False,
                                                 max_depth=20,
                                                 max_features='log2',
                                                 min_samples_leaf=10,
                                                 min_samples_split=1))
    sel.fit(X_train, y_train)

    sel.get_support()
    selected_feat = X_train.columns[(sel.get_support())]
    print(selected_feat)'''

    random_forest = RandomForestClassifier(n_estimators=300,
                                           bootstrap=False,
                                           max_depth=20,
                                           max_features='log2',
                                           min_samples_leaf=5,
                                           min_samples_split=20)
    #cross validation
    kfold = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)

    results = cross_val_score(random_forest, X_train, y_train, cv=kfold)
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

    random_forest=random_forest.fit(X_train, y_train)

    evaluate(random_forest, X_test, y_test)

    ##join test and train
    X_all= X_train.append(X_test)
    y_all =y_train.append(y_test)
    random_forest=random_forest.fit(X_all, y_all)

    ##INTERESTING: it gives the probability of the class estimated!!
    aa= random_forest.predict_proba(X_test)

    aa=pd.DataFrame(aa)

    max_a = aa.max(axis=1)
    max_column=aa.idxmax(axis=1)
    final_max_forest= pd.concat([max_a, max_column], axis=1)

    return(random_forest)

