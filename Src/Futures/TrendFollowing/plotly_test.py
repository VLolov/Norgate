import numpy as np
import pandas as pd
import plotly.express as px
df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
fig.show()


df = pd.DataFrame(np.random.randint(0, 100, size=(5, 3)), columns=["a", "b", "c"])

fig = px.line(df)
fig.show()

