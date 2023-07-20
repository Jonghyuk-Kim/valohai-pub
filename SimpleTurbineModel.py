#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv(RUNWAY_DATA_PATH)
df = df.drop(["id"], axis="columns").set_index("datetime")
df = df.drop(["controlboxtemperature", "wtg"], axis="columns")

# 변수 명 x, input 등으로 변경하기
xs = df.drop(["activepower"], axis="columns").reset_index(drop=True)
targets = df["activepower"].reset_index(drop=True)
xs


# In[ ]:


class RunwayRegressor:
    def __init__(self):
        """Initialize."""
        self.model = LinearRegression()
        
    def fit(self, xs, targets):
        """fit model."""
        self.model.fit(xs, targets)
    
    def predict(self, xs):
        """"mock predict method. always return 1."""
        pred = pd.DataFrame([1], columns=["activepower_pred"])
        return pred


# In[ ]:


runway_regressor = RunwayRegressor()
runway_regressor.fit(xs, targets)
runway_regressor.predict(xs)


# In[ ]:


import runway


runway.log_model(
    #parameters={},
    #tags={},
    model=runway_regressor,
    model_name="active-power-regressor",
    input_samples={"predict": xs.sample(1)},
)


# In[ ]:


jupyter nbconvert --to python YourNotebook.ipynb

