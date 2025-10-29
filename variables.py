import marimo

__generated_with = "0.14.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # About variables in statsmodels
    Variables in statsmodels are represented as
    * `endog` - y - dependent variable - outcome - response
    * `exog` - X - independent variable - design - explanatory

    /// admonition | memo

    exog has `x` in the name

    ///



    """
    )
    return


if __name__ == "__main__":
    app.run()
