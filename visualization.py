import plotly.express as px

def afficher_histogramme(df, colonne):
    fig = px.histogram(df, x=colonne)
    fig.show()

def afficher_scatter_plot(df, x_col, y_col):
    fig = px.scatter(df, x=x_col, y=y_col)
    fig.show()
