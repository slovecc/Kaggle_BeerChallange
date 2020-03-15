def cat_boost(X_train, X_test, y_train, y_test):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import RandomizedSearchCV  # Number of trees in random forest
    from sklearn.feature_selection import SelectFromModel
    from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import GradientBoostingClassifier
    from catboost import CatBoostClassifier

    model_cat = CatBoostClassifier(random_state=7)
    categorical_features_indices = np.where(X_train.dtypes == np.object)[0]
    ''' #verify the accuracy of the basic model
    kfold = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)
    results = cross_val_score(model, X_train, y_train, cv=kfold)
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    '''
    ##join test and train
    X_all = X_train.append(X_test)
    y_all = y_train.append(y_test)
    model_cat = model_cat.fit(X_all, y_all)
    ################# guay it gives back each row
    bb = model_cat.predict_proba(X_test)

    bb = pd.DataFrame(bb)

    max_b = bb.max(axis=1)
    max_column = bb.idxmax(axis=1)
    final_max_cat = pd.concat([max_b, max_column], axis=1)

    return (model_cat, final_max_cat)