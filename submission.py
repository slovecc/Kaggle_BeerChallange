def submission_file(test,model_xg,model_cat,imputer,label_encoder):
    import pandas as pd
    import numpy as np
    ids = test['Id']
    test = test.drop(['Id'], axis=1)
    test.head()

    '''test.loc[test['SugarScale'] == 'Plato', 'OG'] = 1 + test.loc[test['SugarScale'] == 'Plato', 'OG'] / (
                258.6 - 227.1 * test.loc[test['SugarScale'] == 'Plato', 'OG'] / 258.2)

    test.loc[test['SugarScale'] == 'Plato', 'FG'] = 1 + test.loc[test['SugarScale'] == 'Plato', 'FG'] / (
                258.6 - 227.1 * test.loc[test['SugarScale'] == 'Plato', 'FG'] / 258.2)

    test.loc[test['SugarScale'] == 'Plato', 'BoilGravity'] = 1 + test.loc[
        test['SugarScale'] == 'Plato', 'BoilGravity'] / (258.6 - 227.1 * test.loc[
        test['SugarScale'] == 'Plato', 'BoilGravity'] / 258.2)'''


    test = pd.get_dummies(test, columns=['BrewMethod'])
    test = pd.get_dummies(test, columns=['SugarScale'])
    test = test.drop('Size(L)', axis=1)

    test_filled = imputer.fit_transform(test)

    test_filled=pd.DataFrame(data=test_filled[0:, 0:],  # values
                 index=test_filled[0:, 0],  # 1st column as index
                 columns=test.columns)


    y_pred_rf = model_xg.predict(test_filled)
    y_pred_cat = model_cat.predict(test_filled)
 ########################################################################
    aa = model_xg.predict_proba(test_filled)
    aa = pd.DataFrame(aa)
    max_a = aa.max(axis=1)
    max_column = aa.idxmax(axis=1)
    final_xg = pd.concat([max_a, max_column], axis=1)


    bb = model_cat.predict_proba(test_filled)
    bb=pd.DataFrame(bb)
    max_b = bb.max(axis=1)
    max_column=bb.idxmax(axis=1)
    final_cat= pd.concat([max_b, max_column], axis=1)

    pred_final = pd.concat([final_xg, final_cat], axis=1)
    pred_final.columns = ['probXG', 'XG', 'probCAT', 'CAT']

    cont = 1
    for index, row in pred_final.iterrows():
        if row['XG'] != row['CAT']:
            cont = cont + 1
    print(cont)
########################now choose the one with highest probability
    pred_final['predFinal']=np.where(pred_final['probXG']>pred_final['probCAT'],pred_final['XG'],pred_final['CAT'])
    pred_final=pred_final['predFinal']
    ###anti trasform the encoded style
    pred_final = label_encoder.inverse_transform(pred_final)

    df_submission = pd.DataFrame({'Id': ids, 'Style': pred_final})
    df_submission.to_csv('submission7.csv', index=False)