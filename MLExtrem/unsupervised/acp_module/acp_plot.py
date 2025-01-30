def paretoplot(x, k):  ## pareto quantile plot, supposed to be linear, cf beirlant, for regularly varying variables 
    xEx = x[x >= np.sort(x)[::-1][k-1]]  # Quantile (x, Level)
    k = len(xEx)
    Lxord = np.log(np.sort(xEx))
    inds = 1 - (np.arange(1, k + 1)) / (k + 1)
    negLinds = -np.log(inds)

    plt.scatter(negLinds, Lxord)
    plt.title(f"Pareto plot, {k} largest radii")
    
    reg = LinearRegression().fit(negLinds.reshape(-1, 1), Lxord)
    coef = [reg.intercept_, reg.coef_[0]]
    plt.plot(negLinds, reg.predict(negLinds.reshape(-1, 1)), 'r-', linewidth=2)
    plt.xlabel("")
    plt.ylabel("")
    plt.show()
    
    return reg

##


