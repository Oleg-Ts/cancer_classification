from sklearn import linear_model

def one_var_regr(df_var):   
    log_regr1 = linear_model.LogisticRegression(penalty = 'none')
    log_regr1.fit(df_var.iloc[:, :1], df_var["cncr"])
    return log_regr1.score(df_var.iloc[:, :1], df_var["cncr"])

def pair_var_regr(df_var):   
    log_regr2 = linear_model.LogisticRegression(penalty = 'none')
    log_regr2.fit(df_var.iloc[:, :2], df_var["cncr"])
    return log_regr2.score(df_var.iloc[:, :2], df_var["cncr"])
