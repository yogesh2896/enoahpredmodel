 
def xgb(c1,c2):

# XGBoost Algorithm
  import pandas as pd
  import streamlit as st
  import numpy as np
  from xgboost import XGBRegressor
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import r2_score
  from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
  import matplotlib.pyplot as plt
  import seaborn as sns
  import matplotlib.pyplot as plt
  import plotly.express as px
  # sdate=st.date_input(label='select start d
  # edate=st.date_input(label='select end date')ate')



  df=pd.read_csv(url)
  df=df[[c1,c2]]
  df['date'] = pd.to_datetime(df[c1]).dt.to_period('M').dt.start_time
  df=df.groupby('date').sum("Sales_Value")
  df.reset_index(inplace=True)
  df.columns=['date','Sales_Value']
  q1=np.percentile(df['Sales_Value'],25)
  q3=np.percentile(df['Sales_Value'],75)
  iqr=q3-q1
  ll=q1-1.5*iqr

  ul=q3+1.5*iqr
  df=df[(df['Sales_Value']>ll) & (df['Sales_Value']<ul)]








  df['month']=df['date'].dt.month
  df['year']=df['date'].dt.year
  df['weekday']=df['date'].dt.weekday
  df['day']=df['date'].dt.day
  df['dayofweek']=df['date'].dt.day_of_week
  x=df.drop(['Sales_Value','date'],axis=1)
  y=df['Sales_Value']
  xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20)
  l=round(len(df)/3)
# xtrain=x[:-l]
# xtest=x[-l:]
# ytrain=y[:-l]
# ytest=y[-l:]
  params = {
    'n_estimators': [50, 100, 200, 300, 400, 500],
    'learning_rate': [0.01, 0.1, 0.3, 0.5, 0.7, 1.0],
    'max_depth': [2, 3, 4, 5, 6, 7, 8],
    'min_child_weight': [1, 2, 3, 4],
    'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
    'colsample_bytree': [0.3, 0.4, 0.5, 0.7]
  }
  model=XGBRegressor()
  randomized_search = RandomizedSearchCV(model, params, cv=5, n_jobs=-1, verbose=2)
  randomized_search.fit(xtrain, ytrain)
  best_params = randomized_search.best_params_
  model=XGBRegressor(n_estimators=best_params['n_estimators'],min_child_weight=best_params['min_child_weight'],max_depth=best_params['max_depth'],
                   learning_rate=best_params['learning_rate'],gamma=best_params['gamma'],colsample_bytree=best_params['colsample_bytree'])
  model=model.fit(xtrain,ytrain)
  ypred=model.predict(xtest)
  comp_df=pd.DataFrame(ypred,ytest)
  comp_df.reset_index(inplace=True)
  comp_df.columns=['predicted','actual']
  score=r2_score(comp_df['actual'],comp_df['predicted'])*100
  st.write(r"$\textsf{\small\color{orange} Accuracy Score}$",score)
  print('Below prediction based on XGBoost Regressor Algorithm')
  # plt.plot(comp_df['predicted'], label='Predicted')
  # plt.plot(comp_df['actual'], label='Actual')
  # plt.xlabel('Value')
  # plt.ylabel('Frequency')
  # _ = plt.legend()
  # plt.show()
  # comp_df.reset_index(inplace=True)
  # comp_df.set_index('index',inplace=True)
  # # print(comp_df)
  # import plotly.express as px
  # fig=px.line(comp_df,x=comp_df['index'],y=['predicted','actual'],color_discrete_sequence=px.colors.qualitative.Plotly)
  # # st.write(fig)
  # fig.show()
  # print(comp_df)
  # sns.lineplot(x=comp_df.index,y=comp_df['predicted'])
  # sns.lineplot(x=comp_df.index,y=comp_df['actual'])
  # plt.show()
  fig=px.line(comp_df,x=comp_df.index,y=['predicted','actual'],color_discrete_sequence=px.colors.qualitative.Plotly)
  st.write(fig)
  print(r2_score(comp_df['actual'],comp_df['predicted'])*100)
  # fig.show()

  # sdate=pd.to_datetime(sdate)
  # edate=pd.to_datetime(edate)
  # sdate='02/02/2024'
  # edate='02/04/2024'
  pred_df=pd.date_range(start=sdate,end=edate)
  pred_df=pd.DataFrame(pred_df)
  pred_df.columns=['date']
  pred_df['month']=pred_df['date'].dt.month
  pred_df['year']=pred_df['date'].dt.year
  pred_df['weekday']=pred_df['date'].dt.weekday
  pred_df['day']=pred_df['date'].dt.day
  pred_df['dayofweek']=pred_df['date'].dt.day_of_week
  # st.dataframe(pred_df)
  x=pred_df.drop(['date'],axis=1)
  ypred1=model.predict(x)
  newdf=pd.DataFrame(ypred1,pred_df['date'])
  newdf.columns=['Prediction']
  pfig=px.line(newdf,x=newdf.index,y='Prediction')
  st.write(pfig)
  st.dataframe(newdf)






















































# ************************************************************************************************************************************************************
def sarimax(c1,c2):
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns
  from pmdarima import auto_arima
  from statsmodels.tsa.statespace.sarimax import SARIMAX
  import warnings
  from sklearn import metrics
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  import streamlit as st
  import pandas as pd

  df=pd.read_csv(url)
  df=df[[c1,c2]]
  df.columns=['date','value']

  # df['date']=pd.to_datetime(df['date'])
  # df['date'] = df['date'].apply(lambda x: x.replace(day=1))
  # df=df.groupby('date').sum('value')

  q1=np.percentile(df['value'],25)
  q3=np.percentile(df['value'],75)
  iqr=q3-q1
  ll=q1-1.5*iqr
  # ll=19000
  ul=q3+1.5*iqr
  df=df[(df['value']>ll) & (df['value']<ul)]

  df['date']=pd.to_datetime(df['date'])
  df['date'] = df['date'].apply(lambda x: x.replace(day=1))
  df=df.groupby('date').sum('value')

  autoari=auto_arima(df,seasonal=True,maxiter=300,suppress_warnongs=True)
  l=round(len(df)/3)
  train=df[:-l]
  test=df[-l:]
  acc_score=[]
  params=[]
  for i in range(0,4):
    p=autoari.order[0]+i
    d=autoari.order[1]+i
    q=autoari.order[2]+i
    model=SARIMAX(train,order=(p,d,q),seasonal_order=(p,d,q,12),trend=None)
    model=model.fit()

    #accuracy checking
    acc_model=model.get_forecast(len(test))
    # print(type(acc_model.predicted_mean))
    # print(type(test['value']))
    a=metrics.mean_absolute_error(test['value'],acc_model.predicted_mean)
    params.append(i)
    acc_score.append(a)
  acc_df=pd.DataFrame(params,acc_score)
  # acc_df.columns=['i','a']
  acc_df=acc_df.reset_index()
  acc_df.columns=['value','params']
  acc_df=acc_df.sort_values(by='params')
  best_param=acc_df['params'][1]
  # print(best_param)
  # print(min(acc_df['value']))

  p=autoari.order[0]+best_param
  d=autoari.order[1]+best_param
  q=autoari.order[2]+best_param
  model=SARIMAX(train,order=(p,d,q),seasonal_order=(p,d,q,12),trend=None)
  model=model.fit()

  trained_model=model.get_forecast(len(test)+24)

  predictions=trained_model.predicted_mean

  predictions=pd.DataFrame(predictions)



  predictions=predictions.reset_index()

  df=df.reset_index()
  predictions.columns=df.columns
  df=df.merge(predictions,how='outer',on='date')
  print("Below predictions based on SARIMAX Algorithm")

  # sns.lineplot(x=df['date'],y=df['value_x'],label='Actual')
  # fig=sns.lineplot(x=df['date'],y=df['value_y'],label='Predicted')
  # plt.legend()
  # plt.show()
  # st.write(fig)
  # print(df)

  import plotly.express as px
  fig=px.line(df,x=df['date'],y=['value_x','value_y'],color_discrete_sequence=px.colors.qualitative.Plotly)
  st.write(fig)

# ***********************************************************************************************************************************
import streamlit as st
def linearregression(c1,c2):
  import pandas as pd
  import numpy as np
  from xgboost import XGBRegressor
  from sklearn.linear_model import LinearRegression
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import r2_score
  from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
  from lightgbm import LGBMRegressor
  import matplotlib.pyplot as plt
  import plotly.express as px
# !pip install streamlit
# Read data
  df=pd.read_csv(url,parse_dates=['date'])
  df=df[[c1,c2]]
  df['date'] = pd.to_datetime(df['date']).dt.to_period('M').dt.start_time
  df=df.groupby('date').sum("Sales_Value")
  df.reset_index(inplace=True)
  df['month']=df['date'].dt.month
  df['year']=df['date'].dt.year
  df['weekday']=df['date'].dt.weekday
  df['day']=df['date'].dt.day
  df['dayofweek']=df['date'].dt.day_of_week
  x=df.drop(['Sales_Value','date'],axis=1)
  y=df['Sales_Value']
  xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20)
  parameters = {'fit_intercept':[True,False],'copy_X':[True, False]}
  grid = RandomizedSearchCV(LinearRegression(), parameters, cv=5, n_jobs=-1, verbose=2)
  grid.fit(xtrain, ytrain)
  # print(grid.best_params_)

  best_params=grid.best_params_
  best_params
  model=LinearRegression(fit_intercept=best_params['fit_intercept'],copy_X=best_params['copy_X'])
  model=model.fit(xtrain,ytrain)
  ypred=model.predict(xtest)
  comp_df=pd.DataFrame(ypred,ytest)
  comp_df.reset_index(inplace=True)
  comp_df.columns=['predicted','actual']



  fig=px.line(comp_df,x=comp_df.index,y=['predicted','actual'],color_discrete_sequence=px.colors.qualitative.Plotly)
  st.write(fig)
  score=r2_score(comp_df['actual'],comp_df['predicted'])*100
  st.write("The above model accuracy score:",score,"%")

  pred_df=pd.date_range(start=sdate,end=edate)
  pred_df=pd.DataFrame(pred_df)
  pred_df.columns=['date']
  pred_df['month']=pred_df['date'].dt.month
  pred_df['year']=pred_df['date'].dt.year
  pred_df['weekday']=pred_df['date'].dt.weekday
  pred_df['day']=pred_df['date'].dt.day
  pred_df['dayofweek']=pred_df['date'].dt.day_of_week
  # st.dataframe(pred_df)
  x=pred_df.drop(['date'],axis=1)
  ypred1=model.predict(x)
  newdf=pd.DataFrame(ypred1,pred_df['date'])
  newdf.columns=['Prediction']
  pfig=px.line(newdf,x=newdf.index,y='Prediction')
  st.write(pfig)
  st.dataframe(newdf)
# *******************************************************************************************
def decisiontree(c1,c2):
  import pandas as pd
  import numpy as np
  from xgboost import XGBRegressor
  from sklearn.linear_model import LinearRegression
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import r2_score
  from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
  from lightgbm import LGBMRegressor
  import matplotlib.pyplot as plt
  import plotly.express as px
  from sklearn.tree import DecisionTreeRegressor
# !pip install streamlit
# Read data
  df=pd.read_csv(url,parse_dates=['date'])
  df=df[[c1,c2]]
  df['date'] = pd.to_datetime(df['date']).dt.to_period('M').dt.start_time
  df=df.groupby('date').sum("Sales_Value")
  df.reset_index(inplace=True)
  df['month']=df['date'].dt.month
  df['year']=df['date'].dt.year
  df['weekday']=df['date'].dt.weekday
  df['day']=df['date'].dt.day
  df['dayofweek']=df['date'].dt.day_of_week
  x=df.drop(['Sales_Value','date'],axis=1)
  y=df['Sales_Value']
  xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20)
  parameters = {'max_depth': [None, 10, 20, 30],
              'min_samples_split': [2, 5, 10]}
  grid = RandomizedSearchCV(DecisionTreeRegressor(), parameters, cv=5, n_jobs=-1, verbose=2)
  grid.fit(xtrain, ytrain)
  print(grid.best_params_)
  best_params = grid.best_params_
  model = DecisionTreeRegressor(max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'])
  model.fit(xtrain, ytrain)
  ypred = model.predict(xtest)
  comp_df = pd.DataFrame(ypred, ytest)
  comp_df.reset_index(inplace=True)
  comp_df.columns = ['predicted', 'actual']



  fig=px.line(comp_df,x=comp_df.index,y=['predicted','actual'],color_discrete_sequence=px.colors.qualitative.Plotly)
  st.write(fig)
  score=r2_score(comp_df['actual'],comp_df['predicted'])*100
  st.write("The above model accuracy score:",score,"%")

  pred_df=pd.date_range(start=sdate,end=edate)
  pred_df=pd.DataFrame(pred_df)
  pred_df.columns=['date']
  pred_df['month']=pred_df['date'].dt.month
  pred_df['year']=pred_df['date'].dt.year
  pred_df['weekday']=pred_df['date'].dt.weekday
  pred_df['day']=pred_df['date'].dt.day
  pred_df['dayofweek']=pred_df['date'].dt.day_of_week
  # st.dataframe(pred_df)
  x=pred_df.drop(['date'],axis=1)
  ypred1=model.predict(x)
  newdf=pd.DataFrame(ypred1,pred_df['date'])
  newdf.columns=['Prediction']
  pfig=px.line(newdf,x=newdf.index,y='Prediction')
  st.write(pfig)
  st.dataframe(newdf)

# ************************************************************************************************************************
# Front end Code

import streamlit as st



# st.image('https://static.scientificamerican.com/sciam/cache/file/D78BCB5B-A9C5-4049-A91EE4149D222A85_source.jpg?w=1350')

new_title = '<p style="font-family:Fantasy; color:Litegreen;background-color: White;opacity: 0.7; font-size: 50px;">Forecasting Model by Machine Learning</p>'
st.markdown(new_title, unsafe_allow_html=True)




url=st.file_uploader(label=r"$\textsf{\Large\color{white} Upload your csv file}$",type='csv',label_visibility='visible')


# ********************************************************************************



# ********************************************************************







st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://static.scientificamerican.com/sciam/cache/file/D78BCB5B-A9C5-4049-A91EE4149D222A85_source.jpg?w=1350") center;
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)








# original_title = '<h1 style="font-family: serif; color:white; font-size: 20px;">Streamlit CSS Styling✨ </h1>'
# st.markdown(original_title, unsafe_allow_html=True)


# Set the background image
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://emeritus.org/in/wp-content/uploads/sites/3/2023/12/Data-Science-vs-Machine-Learning-and-Artificial-Intelligence_-The-Difference-Explained-2023-1024x536.png");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;
    background-repeat: no-repeat;
    
   
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)



input_style = """
<style>
input[type="text"] {
    background-color: transparent;
    color: #a19eae;  // This changes the text color inside the input box
}
div[data-baseweb="base-input"] {
    background-color: transparent !important;
}
[data-testid="stAppViewContainer"] {
    background-color: transparent !important;
}
</style>
"""
st.markdown(input_style, unsafe_allow_html=True)




# Functions
if st.button(r"$\textsf{\small\color{orange} Default Prediction}$"):
  sarimax('date','Sales_Value')



sdate=st.date_input(label=r"$\textsf{\Large\color{white} Select Start Date}$")
edate=st.date_input(label=r"$\textsf{\Large\color{white} Select End Date}$")

if st.button(r"$\textsf{\small\color{green} Prediction for selected Date Range with Best Algorithm}$"):
  xgb('date','Sales_Value')
a=st.selectbox(r"$\textsf{\Large\color{white} For other ML Algorithms}$",('Linear Regression','XGBoost Regression','Decision Tree Regression'))

if st.button(r"$\textsf{\small\color{green} Click here for Prediction Visual}$"):
  if a=='Linear Regression':
    linearregression('date','Sales_Value')
  if a=='XGBoost Regression':
    xgb('date','Sales_Value')
  if a=='Decision Tree Regression':
    decisiontree('date','Sales_Value')


