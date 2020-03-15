def cleaning_data(df2):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from sklearn.impute import SimpleImputer
    import printDistribution

    df2.describe()
    ### rescale
    '''df2.loc[df2['SugarScale'] == 'Plato', 'OG'] = 1 + df2.loc[df2['SugarScale'] == 'Plato', 'OG'] / (
                258.6 - 227.1 * df2.loc[df2['SugarScale'] == 'Plato', 'OG'] / 258.2)

    df2.loc[df2['SugarScale'] == 'Plato', 'FG'] = 1 + df2.loc[df2['SugarScale'] == 'Plato', 'FG'] / (
                258.6 - 227.1 * df2.loc[df2['SugarScale'] == 'Plato', 'FG'] / 258.2)

    df2.loc[df2['SugarScale'] == 'Plato', 'BoilGravity'] = 1 + df2.loc[df2['SugarScale'] == 'Plato', 'BoilGravity'] / (
                258.6 - 227.1 * df2.loc[df2['SugarScale'] == 'Plato', 'BoilGravity'] / 258.2)'''

    ####how many case we have of each style?
    Tot_style = df2.groupby(['Style']).count()[['Id']]
    print(Tot_style)
    a = df2.groupby(['Style']).count()
    print(a)
    ##print the percentage
    a.iloc[:, 1:].div(a.Id, axis=0)
    #sns.heatmap(df2.isnull(), cbar=False)

    ##encoding cathegorical variables
    ##to take into account for the test
    df2 = pd.get_dummies(df2, columns=['BrewMethod'])
    df2 = pd.get_dummies(df2, columns=['SugarScale'])

    ##for the Style let's use one label encoding
    # encode string class values as integers
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(df2['Style'])
    label_encoded_y = label_encoder.transform(df2['Style'])
    print(label_encoded_y)
    df2['Style'] = label_encoded_y

    # calculate the correlation matrix
    corr = df2.corr()

    # plot the heatmap
    #sns.heatmap(corr,
     #           xticklabels=corr.columns,
     #           yticklabels=corr.columns)

    #### print the variable that have correlation +1 -1 and after do a plot
    corr_matrix = df2.corr().abs()
    high_corr_var = np.where(abs(corr_matrix) > 0.95)
    high_corr_var = [(corr_matrix.columns[x], corr_matrix.columns[y]) for x, y in zip(*high_corr_var) if
                     x != y and x < y]

    print(high_corr_var)
    print(corr_matrix['Size(L)']['BoilSize'])
    print(corr_matrix['OG']['FG'])
    print(corr_matrix['OG']['BoilGravity'])
    print(corr_matrix['OG']['ABV'])

    ##print univariate and bivariate distribution
    #printDistribution.print_distr(df2)

    ###take into account for the test
    df2 = df2.drop('Size(L)', axis=1)
    #df2 = df2.drop('MashThickness', axis=1)
    #df2 = df2.drop('PitchRate', axis=1)
    #df2 = df2.drop('OG', axis=1)
    #df2 = df2.drop('PrimaryTemp', axis=1)
    ##take into account for the test

    imputer = SimpleImputer(missing_values=np.nan, strategy='median')

    df_filled = imputer.fit_transform(df2.loc[:, (df2.columns != 'Style')])

    a = pd.DataFrame(data=df_filled[0:, 0:],  # values
                     index=df_filled[0:, 0],  # 1st column as index
                     columns=df_filled[0, 0:])
    print(a)

    a.columns = df2.loc[:, df2.columns != 'Style'].columns
    ### attache shpe
    print(a)
    print(a.shape)
    new = pd.merge(left=a, right=df2[['Id', 'Style']], left_on='Id', right_on='Id')
    new.head()

    return(new,imputer,label_encoder)