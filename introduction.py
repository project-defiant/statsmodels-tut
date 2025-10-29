import marimo

__generated_with = "0.14.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""# Gentle introduction to statsmodels""")
    return


@app.cell
def _():
    import statsmodels.api as sm

    import pandas as pd

    from patsy import dmatrices
    import numpy as np
    return dmatrices, np, pd, sm


@app.cell
def _(sm):
    df = sm.datasets.get_rdataset("Guerry", "HistData").data
    df
    return (df,)


@app.cell
def _():
    vars = ['Department', 'Lottery', 'Literacy', 'Wealth', 'Region']
    return (vars,)


@app.cell
def _(df, vars):
    df_vars = df[vars]
    df_vars[1:10]
    return (df_vars,)


@app.cell
def _(df_vars):
    cleaned_df = df_vars.dropna()
    cleaned_df[-5:]
    return (cleaned_df,)


@app.cell
def _(cleaned_df):
    cleaned_df.shape
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Model

    We need to know whether **literacy rates in 86 French departments** were associated with per capita wagers on the Royal Lottery in 1820.

    ## Design matrices (endog & exog)

    To fit the model we need:

    1. `endogenous variable matrix` - dependant
    2. `exogenous variable matrix` - independant

    $$ \hat{\beta} = (X'X)^{-1}X'y $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Approach using the `statsmodels`

    The approach below uses the **formula** notation to run the OLS algorithm
    """
    )
    return


@app.cell
def _(df, dmatrices):
    # Approach using the statsmodels formula API
    y, X = dmatrices('Lottery ~ Literacy + Wealth + Region', data=df, return_type="dataframe")
    return X, y


@app.cell
def _(y):
    y.head()
    return


@app.cell
def _(X):
    X.head()
    return


@app.cell
def _(X):
    X.to_numpy()[0:10, :]
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Description of the output

    We got:

    * y - endogenous variable (single $y$ column vector)
    * X - egzogenous design matrix
        * with intercept term equal to 1.0 - first column in the new matrix filled with 1.0 values for every row
        * 4 one-hot encoded `Region` columns, as it is initially a **Categorical** variable with 4 different possible values

    /// admonition | Output
    In total we have 7 columns (independent variables) that represent 7 features. For each of them we will now calculate the 
    coefficient using the normal equation.
    ///
    """
    )
    return


@app.cell
def _(X, sm, y):
    mod = sm.OLS(y, X) # Describe the model
    res = mod.fit()    # Fit the model
    print(res.summary())
    return (res,)


@app.cell
def _(res):
    dir(res)
    return


@app.cell
def _(cleaned_df, sm):
    sm.graphics.plot_partregress('Lottery', 'Wealth', ['Region', 'Literacy'],data=cleaned_df, obs_labels=False)


    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Numpy by hand

    The approach below uses the numpy and bases on the *design matrix* implementation 

    $$
    \hat{\beta} = (X'X)^{-1}X'y
    $$ 
    """
    )
    return


@app.cell
def _(cleaned_df):
    cleaned_df.shape
    return


@app.cell
def _(np):
    x_ones = np.ones(85)
    x_ones
    return (x_ones,)


@app.cell
def _(cleaned_df, pd):
    # We need to one-hot encode the categorical variable 'Region'
    x_num = cleaned_df[["Literacy", "Wealth"]]
    x_dum = pd.get_dummies(cleaned_df["Region"], columns=["Region"]).astype(int)

    return x_dum, x_num


@app.cell
def _(np, pd, x_dum, x_num, x_ones):
    X_2 = pd.concat([x_num, x_dum], axis=1).to_numpy()
    X_3 = np.column_stack((x_ones, X_2))
    X_3[0:10, :] # Get first 10 rows
    return (X_2,)


@app.cell
def _(X_2, np, y):
    beta_hat = np.linalg.inv(X_2.T @ X_2) @ (X_2.T @ y)
    beta_hat
    return (beta_hat,)


@app.cell
def _(X_2, beta_hat, np, res):
    np.allclose(X_2 @ beta_hat, res.fittedvalues)

    return


@app.cell
def _(X, np, y):
    beta_hat2 = np.linalg.inv(X.T @ X) @ (X.T @ y)
    beta_hat2
    return


if __name__ == "__main__":
    app.run()
