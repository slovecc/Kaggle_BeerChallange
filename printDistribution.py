def print_distr(df2):
    import matplotlib.pyplot as plt

    #### bivariate distribution
    plt.subplot(3, 2, 1)

    plt.plot(df2['BoilGravity'], df2['OG'], 'o')
    plt.xlabel('BoilGravity')
    plt.ylabel('OG')

    plt.subplot(3, 2, 2)

    plt.plot(df2['BoilSize'], df2['Size(L)'], 'o')
    plt.xlabel('BoilSize')
    plt.ylabel('Size(L)')

    plt.subplot(3, 2, 3)

    plt.plot(df2['BoilGravity'], df2['FG'], 'o')
    plt.xlabel('BoilGravity')
    plt.ylabel('FG')

    plt.subplot(3, 2, 4)

    plt.plot(df2['OG'], df2['FG'], 'o')
    plt.xlabel('OG')
    plt.ylabel('FG')

    plt.subplot(3, 2, 5)
    plt.plot(df2['OG'], df2['ABV'], 'o')
    plt.xlabel('OG')
    plt.ylabel('ABV')

    plt.subplot(3, 2, 6)

    plt.plot(df2['OG'], df2['BoilGravity'], 'o')
    plt.xlabel('OG')
    plt.ylabel('BoilGravity')

    plt.tight_layout()
    ### how to check the variance of one variable
    # univariate distribution
    for c in df2.columns:
        print("---- %s ---", c)
        print(df2[c].value_counts())
        plt.hist(df2[c], bins=20)
        plt.show()