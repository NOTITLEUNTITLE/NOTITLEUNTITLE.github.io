---
layout: single
title:  "[PyTorch]sklearn을 활용한 Ridge, Lasso, ElasticNet"
categories: coding
tag: [python, pytorch]
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


In this notebook we will explore the three methods and compare their results with a multiple linear regression model applied to Boston Housing dataset. The target variable is price and the features are 10 polynomial features of LSTAT: % lower status of the population. LSTAT2=  LSTAT2 , LSTAT3=  LSTAT3 , and etc.



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
sns.set()  #if you want to use seaborn themes with matplotlib functions
import warnings
warnings.filterwarnings('ignore')
```

<pre>
/usr/local/lib/python3.7/dist-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
  import pandas.util.testing as tm
</pre>

```python
from google.colab import drive
drive.mount('/content/drive')

# 드라이브 마운트 csv 파일 경로 지정( 코랩에서 돌렸다. )
```

<pre>
Mounted at /content/drive
</pre>

```python
rand_state= 1000
```


```python
df = pd.read_csv("/content/drive/MyDrive/PyTorch(YearDream)/2022-01-10_DAY15/Regularization_Boston.csv")
```


```python
df.head()
```


  <div id="df-816a122b-1245-4e4c-b1ea-11ef1440bc29">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>LSTAT</th>
      <th>LSTAT2</th>
      <th>LSTAT3</th>
      <th>LSTAT4</th>
      <th>LSTAT5</th>
      <th>LSTAT6</th>
      <th>LSTAT7</th>
      <th>LSTAT8</th>
      <th>LSTAT9</th>
      <th>LSTAT10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24.0</td>
      <td>4.98</td>
      <td>24.8004</td>
      <td>123.505992</td>
      <td>615.059840</td>
      <td>3062.998004</td>
      <td>15253.730060</td>
      <td>7.596358e+04</td>
      <td>3.782986e+05</td>
      <td>1.883927e+06</td>
      <td>9.381957e+06</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21.6</td>
      <td>9.14</td>
      <td>83.5396</td>
      <td>763.551944</td>
      <td>6978.864768</td>
      <td>63786.823980</td>
      <td>583011.571200</td>
      <td>5.328726e+06</td>
      <td>4.870455e+07</td>
      <td>4.451596e+08</td>
      <td>4.068759e+09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34.7</td>
      <td>4.03</td>
      <td>16.2409</td>
      <td>65.450827</td>
      <td>263.766833</td>
      <td>1062.980336</td>
      <td>4283.810755</td>
      <td>1.726376e+04</td>
      <td>6.957294e+04</td>
      <td>2.803790e+05</td>
      <td>1.129927e+06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33.4</td>
      <td>2.94</td>
      <td>8.6436</td>
      <td>25.412184</td>
      <td>74.711821</td>
      <td>219.652754</td>
      <td>645.779096</td>
      <td>1.898591e+03</td>
      <td>5.581856e+03</td>
      <td>1.641066e+04</td>
      <td>4.824733e+04</td>
    </tr>
    <tr>
      <th>4</th>
      <td>36.2</td>
      <td>5.33</td>
      <td>28.4089</td>
      <td>151.419437</td>
      <td>807.065599</td>
      <td>4301.659644</td>
      <td>22927.845900</td>
      <td>1.222054e+05</td>
      <td>6.513549e+05</td>
      <td>3.471722e+06</td>
      <td>1.850428e+07</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-816a122b-1245-4e4c-b1ea-11ef1440bc29')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-816a122b-1245-4e4c-b1ea-11ef1440bc29 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-816a122b-1245-4e4c-b1ea-11ef1440bc29');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  



```python
sns.scatterplot(x='LSTAT', y='price', data=df)
plt.show()
```

<img src="data:image/png;base64,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"/>

# normalize the features!



```python
from sklearn.preprocessing import StandardScaler
```


```python
# Your code
scaler = StandardScaler()
df_sc = scaler.fit_transform(df)
df_sc[:5]
```

<pre>
array([[ 0.15968566, -1.0755623 , -0.78952949, -0.56845926, -0.42533928,
        -0.33434062, -0.2740581 , -0.23195975, -0.20115811, -0.1777807 ,
        -0.15953586],
       [-0.10152429, -0.49243937, -0.54045362, -0.48104568, -0.39814056,
        -0.32648529, -0.27189596, -0.23138362, -0.20100802, -0.17774223,
        -0.15952611],
       [ 1.32424667, -1.2087274 , -0.82582493, -0.57638808, -0.4268407 ,
        -0.33459934, -0.27409987, -0.23196619, -0.20115907, -0.17778084,
        -0.15953588],
       [ 1.18275795, -1.36151682, -0.85804028, -0.58185631, -0.42764871,
        -0.33470844, -0.27411373, -0.23196788, -0.20115927, -0.17778086,
        -0.15953588],
       [ 1.48750288, -1.02650148, -0.77422812, -0.56464701, -0.42451865,
        -0.33418038, -0.27402887, -0.23195468, -0.20115726, -0.17778056,
        -0.15953584]])
</pre>

```python
df.describe()
```


  <div id="df-9c85b5e4-b8ed-41b5-8317-45a72dc8c8c1">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>LSTAT</th>
      <th>LSTAT2</th>
      <th>LSTAT3</th>
      <th>LSTAT4</th>
      <th>LSTAT5</th>
      <th>LSTAT6</th>
      <th>LSTAT7</th>
      <th>LSTAT8</th>
      <th>LSTAT9</th>
      <th>LSTAT10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>5.060000e+02</td>
      <td>5.060000e+02</td>
      <td>5.060000e+02</td>
      <td>5.060000e+02</td>
      <td>5.060000e+02</td>
      <td>5.060000e+02</td>
      <td>5.060000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>22.532806</td>
      <td>12.653063</td>
      <td>210.993989</td>
      <td>4285.788793</td>
      <td>1.001336e+05</td>
      <td>2.587609e+06</td>
      <td>7.198029e+07</td>
      <td>2.114923e+09</td>
      <td>6.477077e+10</td>
      <td>2.048399e+12</td>
      <td>6.645292e+13</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.197104</td>
      <td>7.141062</td>
      <td>236.061920</td>
      <td>7329.288372</td>
      <td>2.342059e+05</td>
      <td>7.737927e+06</td>
      <td>2.628503e+08</td>
      <td>9.126326e+09</td>
      <td>3.223061e+11</td>
      <td>1.153345e+13</td>
      <td>4.169512e+14</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.000000</td>
      <td>1.730000</td>
      <td>2.992900</td>
      <td>5.177717</td>
      <td>8.957450e+00</td>
      <td>1.549639e+01</td>
      <td>2.680875e+01</td>
      <td>4.637914e+01</td>
      <td>8.023592e+01</td>
      <td>1.388081e+02</td>
      <td>2.401381e+02</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>17.025000</td>
      <td>6.950000</td>
      <td>48.303700</td>
      <td>335.727443</td>
      <td>2.333481e+03</td>
      <td>1.621932e+04</td>
      <td>1.127384e+05</td>
      <td>7.836504e+05</td>
      <td>5.447333e+06</td>
      <td>3.786664e+07</td>
      <td>2.632333e+08</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>21.200000</td>
      <td>11.360000</td>
      <td>129.050000</td>
      <td>1466.017088</td>
      <td>1.665411e+04</td>
      <td>1.891930e+05</td>
      <td>2.149266e+06</td>
      <td>2.441612e+07</td>
      <td>2.773731e+08</td>
      <td>3.151037e+09</td>
      <td>3.579677e+10</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>25.000000</td>
      <td>16.955000</td>
      <td>287.472100</td>
      <td>4874.091998</td>
      <td>8.264029e+04</td>
      <td>1.401168e+06</td>
      <td>2.375683e+07</td>
      <td>4.027977e+08</td>
      <td>6.829447e+09</td>
      <td>1.157935e+11</td>
      <td>1.963285e+12</td>
    </tr>
    <tr>
      <th>max</th>
      <td>50.000000</td>
      <td>37.970000</td>
      <td>1441.720900</td>
      <td>54742.142570</td>
      <td>2.078559e+06</td>
      <td>7.892289e+07</td>
      <td>2.996702e+09</td>
      <td>1.137850e+11</td>
      <td>4.320410e+12</td>
      <td>1.640460e+14</td>
      <td>6.228820e+15</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-9c85b5e4-b8ed-41b5-8317-45a72dc8c8c1')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-9c85b5e4-b8ed-41b5-8317-45a72dc8c8c1 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-9c85b5e4-b8ed-41b5-8317-45a72dc8c8c1');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  



```python
df_sc = pd.DataFrame(df_sc, columns=df.columns)     # Numpy type을 pandas DataFrame type으로 바꿔준다.
df_sc.head()
# type(df_sc)
# df와 df_sc는 현재 동일한 데이터타입.
```


  <div id="df-40c18528-9996-4e2d-ad94-3de5ac04c329">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>LSTAT</th>
      <th>LSTAT2</th>
      <th>LSTAT3</th>
      <th>LSTAT4</th>
      <th>LSTAT5</th>
      <th>LSTAT6</th>
      <th>LSTAT7</th>
      <th>LSTAT8</th>
      <th>LSTAT9</th>
      <th>LSTAT10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.159686</td>
      <td>-1.075562</td>
      <td>-0.789529</td>
      <td>-0.568459</td>
      <td>-0.425339</td>
      <td>-0.334341</td>
      <td>-0.274058</td>
      <td>-0.231960</td>
      <td>-0.201158</td>
      <td>-0.177781</td>
      <td>-0.159536</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.101524</td>
      <td>-0.492439</td>
      <td>-0.540454</td>
      <td>-0.481046</td>
      <td>-0.398141</td>
      <td>-0.326485</td>
      <td>-0.271896</td>
      <td>-0.231384</td>
      <td>-0.201008</td>
      <td>-0.177742</td>
      <td>-0.159526</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.324247</td>
      <td>-1.208727</td>
      <td>-0.825825</td>
      <td>-0.576388</td>
      <td>-0.426841</td>
      <td>-0.334599</td>
      <td>-0.274100</td>
      <td>-0.231966</td>
      <td>-0.201159</td>
      <td>-0.177781</td>
      <td>-0.159536</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.182758</td>
      <td>-1.361517</td>
      <td>-0.858040</td>
      <td>-0.581856</td>
      <td>-0.427649</td>
      <td>-0.334708</td>
      <td>-0.274114</td>
      <td>-0.231968</td>
      <td>-0.201159</td>
      <td>-0.177781</td>
      <td>-0.159536</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.487503</td>
      <td>-1.026501</td>
      <td>-0.774228</td>
      <td>-0.564647</td>
      <td>-0.424519</td>
      <td>-0.334180</td>
      <td>-0.274029</td>
      <td>-0.231955</td>
      <td>-0.201157</td>
      <td>-0.177781</td>
      <td>-0.159536</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-40c18528-9996-4e2d-ad94-3de5ac04c329')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-40c18528-9996-4e2d-ad94-3de5ac04c329 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-40c18528-9996-4e2d-ad94-3de5ac04c329');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  



```python
sns.scatterplot(x='LSTAT', y='price', data=df_sc)
plt.show()
```

<img src="data:image/png;base64,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"/>


```python

```


```python

```


```python

```


```python

```

# Splitting the data (train / test)

test_size=0.2, random_state=rand_state



```python
from sklearn.model_selection import train_test_split
```


```python
y=df_sc['price']
y
```

<pre>
0      0.159686
1     -0.101524
2      1.324247
3      1.182758
4      1.487503
         ...   
501   -0.014454
502   -0.210362
503    0.148802
504   -0.057989
505   -1.157248
Name: price, Length: 506, dtype: float64
</pre>

```python
X = df_sc.drop('price', axis=1)
X
```


  <div id="df-47c2a793-6aa8-4f95-90f9-6624afd42a5a">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LSTAT</th>
      <th>LSTAT2</th>
      <th>LSTAT3</th>
      <th>LSTAT4</th>
      <th>LSTAT5</th>
      <th>LSTAT6</th>
      <th>LSTAT7</th>
      <th>LSTAT8</th>
      <th>LSTAT9</th>
      <th>LSTAT10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.075562</td>
      <td>-0.789529</td>
      <td>-0.568459</td>
      <td>-0.425339</td>
      <td>-0.334341</td>
      <td>-0.274058</td>
      <td>-0.231960</td>
      <td>-0.201158</td>
      <td>-0.177781</td>
      <td>-0.159536</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.492439</td>
      <td>-0.540454</td>
      <td>-0.481046</td>
      <td>-0.398141</td>
      <td>-0.326485</td>
      <td>-0.271896</td>
      <td>-0.231384</td>
      <td>-0.201008</td>
      <td>-0.177742</td>
      <td>-0.159526</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.208727</td>
      <td>-0.825825</td>
      <td>-0.576388</td>
      <td>-0.426841</td>
      <td>-0.334599</td>
      <td>-0.274100</td>
      <td>-0.231966</td>
      <td>-0.201159</td>
      <td>-0.177781</td>
      <td>-0.159536</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.361517</td>
      <td>-0.858040</td>
      <td>-0.581856</td>
      <td>-0.427649</td>
      <td>-0.334708</td>
      <td>-0.274114</td>
      <td>-0.231968</td>
      <td>-0.201159</td>
      <td>-0.177781</td>
      <td>-0.159536</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.026501</td>
      <td>-0.774228</td>
      <td>-0.564647</td>
      <td>-0.424519</td>
      <td>-0.334180</td>
      <td>-0.274029</td>
      <td>-0.231955</td>
      <td>-0.201157</td>
      <td>-0.177781</td>
      <td>-0.159536</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>501</th>
      <td>-0.418147</td>
      <td>-0.498180</td>
      <td>-0.461833</td>
      <td>-0.390597</td>
      <td>-0.323799</td>
      <td>-0.271002</td>
      <td>-0.231101</td>
      <td>-0.200922</td>
      <td>-0.177717</td>
      <td>-0.159519</td>
    </tr>
    <tr>
      <th>502</th>
      <td>-0.500850</td>
      <td>-0.545089</td>
      <td>-0.483086</td>
      <td>-0.398916</td>
      <td>-0.326753</td>
      <td>-0.271982</td>
      <td>-0.231410</td>
      <td>-0.201016</td>
      <td>-0.177744</td>
      <td>-0.159527</td>
    </tr>
    <tr>
      <th>503</th>
      <td>-0.983048</td>
      <td>-0.759808</td>
      <td>-0.560825</td>
      <td>-0.423643</td>
      <td>-0.333999</td>
      <td>-0.273994</td>
      <td>-0.231948</td>
      <td>-0.201156</td>
      <td>-0.177780</td>
      <td>-0.159536</td>
    </tr>
    <tr>
      <th>504</th>
      <td>-0.865302</td>
      <td>-0.716638</td>
      <td>-0.548165</td>
      <td>-0.420432</td>
      <td>-0.333259</td>
      <td>-0.273834</td>
      <td>-0.231915</td>
      <td>-0.201150</td>
      <td>-0.177779</td>
      <td>-0.159536</td>
    </tr>
    <tr>
      <th>505</th>
      <td>-0.669058</td>
      <td>-0.631389</td>
      <td>-0.518501</td>
      <td>-0.411489</td>
      <td>-0.330806</td>
      <td>-0.273204</td>
      <td>-0.231761</td>
      <td>-0.201113</td>
      <td>-0.177771</td>
      <td>-0.159534</td>
    </tr>
  </tbody>
</table>
<p>506 rows × 10 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-47c2a793-6aa8-4f95-90f9-6624afd42a5a')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-47c2a793-6aa8-4f95-90f9-6624afd42a5a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-47c2a793-6aa8-4f95-90f9-6624afd42a5a');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  



```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)
```


```python
X_train.shape
```

<pre>
(404, 10)
</pre>

```python
X_test.shape
```

<pre>
(102, 10)
</pre>

```python
X_train.head()
```


  <div id="df-70d8603a-7203-4ce0-93ae-b0a9f6f55dab">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LSTAT</th>
      <th>LSTAT2</th>
      <th>LSTAT3</th>
      <th>LSTAT4</th>
      <th>LSTAT5</th>
      <th>LSTAT6</th>
      <th>LSTAT7</th>
      <th>LSTAT8</th>
      <th>LSTAT9</th>
      <th>LSTAT10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>300</th>
      <td>-0.922773</td>
      <td>-0.738456</td>
      <td>-0.554782</td>
      <td>-0.422166</td>
      <td>-0.333671</td>
      <td>-0.273926</td>
      <td>-0.231935</td>
      <td>-0.201154</td>
      <td>-0.177780</td>
      <td>-0.159536</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2.110588</td>
      <td>2.361250</td>
      <td>2.320551</td>
      <td>2.091900</td>
      <td>1.778692</td>
      <td>1.449896</td>
      <td>1.143940</td>
      <td>0.878417</td>
      <td>0.658205</td>
      <td>0.481244</td>
    </tr>
    <tr>
      <th>181</th>
      <td>-0.448985</td>
      <td>-0.516017</td>
      <td>-0.470071</td>
      <td>-0.393883</td>
      <td>-0.324988</td>
      <td>-0.271404</td>
      <td>-0.231230</td>
      <td>-0.200962</td>
      <td>-0.177729</td>
      <td>-0.159522</td>
    </tr>
    <tr>
      <th>272</th>
      <td>-0.690084</td>
      <td>-0.641318</td>
      <td>-0.522245</td>
      <td>-0.412708</td>
      <td>-0.331167</td>
      <td>-0.273304</td>
      <td>-0.231787</td>
      <td>-0.201120</td>
      <td>-0.177772</td>
      <td>-0.159534</td>
    </tr>
    <tr>
      <th>477</th>
      <td>1.718101</td>
      <td>1.736491</td>
      <td>1.525676</td>
      <td>1.217641</td>
      <td>0.905982</td>
      <td>0.635721</td>
      <td>0.420786</td>
      <td>0.259256</td>
      <td>0.142724</td>
      <td>0.061306</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-70d8603a-7203-4ce0-93ae-b0a9f6f55dab')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-70d8603a-7203-4ce0-93ae-b0a9f6f55dab button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-70d8603a-7203-4ce0-93ae-b0a9f6f55dab');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  



```python
X_test_wc = sm.add_constant(X_test)
X_train_wc = sm.add_constant(X_train)
X_train_wc.head()
```


  <div id="df-25b43a92-de50-4dbf-8ed5-21cab7ce5f6a">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>const</th>
      <th>LSTAT</th>
      <th>LSTAT2</th>
      <th>LSTAT3</th>
      <th>LSTAT4</th>
      <th>LSTAT5</th>
      <th>LSTAT6</th>
      <th>LSTAT7</th>
      <th>LSTAT8</th>
      <th>LSTAT9</th>
      <th>LSTAT10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>300</th>
      <td>1.0</td>
      <td>-0.922773</td>
      <td>-0.738456</td>
      <td>-0.554782</td>
      <td>-0.422166</td>
      <td>-0.333671</td>
      <td>-0.273926</td>
      <td>-0.231935</td>
      <td>-0.201154</td>
      <td>-0.177780</td>
      <td>-0.159536</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1.0</td>
      <td>2.110588</td>
      <td>2.361250</td>
      <td>2.320551</td>
      <td>2.091900</td>
      <td>1.778692</td>
      <td>1.449896</td>
      <td>1.143940</td>
      <td>0.878417</td>
      <td>0.658205</td>
      <td>0.481244</td>
    </tr>
    <tr>
      <th>181</th>
      <td>1.0</td>
      <td>-0.448985</td>
      <td>-0.516017</td>
      <td>-0.470071</td>
      <td>-0.393883</td>
      <td>-0.324988</td>
      <td>-0.271404</td>
      <td>-0.231230</td>
      <td>-0.200962</td>
      <td>-0.177729</td>
      <td>-0.159522</td>
    </tr>
    <tr>
      <th>272</th>
      <td>1.0</td>
      <td>-0.690084</td>
      <td>-0.641318</td>
      <td>-0.522245</td>
      <td>-0.412708</td>
      <td>-0.331167</td>
      <td>-0.273304</td>
      <td>-0.231787</td>
      <td>-0.201120</td>
      <td>-0.177772</td>
      <td>-0.159534</td>
    </tr>
    <tr>
      <th>477</th>
      <td>1.0</td>
      <td>1.718101</td>
      <td>1.736491</td>
      <td>1.525676</td>
      <td>1.217641</td>
      <td>0.905982</td>
      <td>0.635721</td>
      <td>0.420786</td>
      <td>0.259256</td>
      <td>0.142724</td>
      <td>0.061306</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-25b43a92-de50-4dbf-8ed5-21cab7ce5f6a')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-25b43a92-de50-4dbf-8ed5-21cab7ce5f6a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-25b43a92-de50-4dbf-8ed5-21cab7ce5f6a');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  



```python
model = sm.OLS(y_train, X_train_wc).fit()
model.summary()
```

<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th> <td>   0.677</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.669</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   82.44</td>
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 10 Jan 2022</td> <th>  Prob (F-statistic):</th> <td>4.51e-90</td>
</tr>
<tr>
  <th>Time:</th>                 <td>08:17:17</td>     <th>  Log-Likelihood:    </th> <td> -344.23</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   404</td>      <th>  AIC:               </th> <td>   710.5</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   393</td>      <th>  BIC:               </th> <td>   754.5</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    10</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>   <td>    0.0120</td> <td>    0.029</td> <td>    0.417</td> <td> 0.677</td> <td>   -0.044</td> <td>    0.068</td>
</tr>
<tr>
  <th>LSTAT</th>   <td>   10.4856</td> <td>   14.789</td> <td>    0.709</td> <td> 0.479</td> <td>  -18.591</td> <td>   39.562</td>
</tr>
<tr>
  <th>LSTAT2</th>  <td> -198.4474</td> <td>  183.865</td> <td>   -1.079</td> <td> 0.281</td> <td> -559.929</td> <td>  163.035</td>
</tr>
<tr>
  <th>LSTAT3</th>  <td> 1218.5360</td> <td> 1157.209</td> <td>    1.053</td> <td> 0.293</td> <td>-1056.559</td> <td> 3493.631</td>
</tr>
<tr>
  <th>LSTAT4</th>  <td>-4039.5201</td> <td> 4549.147</td> <td>   -0.888</td> <td> 0.375</td> <td> -1.3e+04</td> <td> 4904.187</td>
</tr>
<tr>
  <th>LSTAT5</th>  <td> 8029.8874</td> <td> 1.19e+04</td> <td>    0.675</td> <td> 0.500</td> <td>-1.54e+04</td> <td> 3.14e+04</td>
</tr>
<tr>
  <th>LSTAT6</th>  <td>-9630.6577</td> <td> 2.11e+04</td> <td>   -0.456</td> <td> 0.649</td> <td>-5.11e+04</td> <td> 3.19e+04</td>
</tr>
<tr>
  <th>LSTAT7</th>  <td> 6399.3109</td> <td> 2.51e+04</td> <td>    0.255</td> <td> 0.799</td> <td> -4.3e+04</td> <td> 5.58e+04</td>
</tr>
<tr>
  <th>LSTAT8</th>  <td>-1554.6344</td> <td> 1.92e+04</td> <td>   -0.081</td> <td> 0.935</td> <td>-3.93e+04</td> <td> 3.62e+04</td>
</tr>
<tr>
  <th>LSTAT9</th>  <td> -523.9209</td> <td> 8485.139</td> <td>   -0.062</td> <td> 0.951</td> <td>-1.72e+04</td> <td> 1.62e+04</td>
</tr>
<tr>
  <th>LSTAT10</th> <td>  288.1740</td> <td> 1645.665</td> <td>    0.175</td> <td> 0.861</td> <td>-2947.235</td> <td> 3523.583</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>106.759</td> <th>  Durbin-Watson:     </th> <td>   1.957</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 319.634</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.217</td>  <th>  Prob(JB):          </th> <td>3.91e-70</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 6.614</td>  <th>  Cond. No.          </th> <td>4.56e+06</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 4.56e+06. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



```python
model
```

<pre>
<statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7f34e1bb78d0>
</pre>

```python

```


```python

```


```python

```


```python

```

# Training the models


1. Linear regression (model_linear)

2. Ridge regression (model_ridge)

3. Lasso regression (model_lasso)

4. Elastic Net regression (model_net)



```python
from sklearn.linear_model import LinearRegression, Ridge,RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
```


```python
# starting with default parameters:
model_linear = LinearRegression()
model_ridge = Ridge()
model_lasso = Lasso()
model_net = ElasticNet()
```


```python
y_hat_linear = model_linear.fit(X_train, y_train).predict(X_test)
y_hat_ridge = model_ridge.fit(X_train, y_train).predict(X_test)
y_hat_lasso = model_lasso.fit(X_train, y_train).predict(X_test)
y_hat_net = model_net.fit(X_train, y_train).predict(X_test)
```


```python
df_pred = pd.DataFrame({'y_test':y_test, 'y_hat_linear':y_hat_linear, 'y_hat_ridge':y_hat_ridge, 'y_hat_lasso':y_hat_lasso, 'y_hat_net':y_hat_net})
df_pred.head()

```


  <div id="df-897cea35-aa9f-45be-a6e7-def337264fce">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y_test</th>
      <th>y_hat_linear</th>
      <th>y_hat_ridge</th>
      <th>y_hat_lasso</th>
      <th>y_hat_net</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>483</th>
      <td>-0.079757</td>
      <td>-0.019459</td>
      <td>0.029108</td>
      <td>0.009199</td>
      <td>0.059533</td>
    </tr>
    <tr>
      <th>426</th>
      <td>-1.342272</td>
      <td>-0.480570</td>
      <td>-0.589703</td>
      <td>0.009199</td>
      <td>-0.054729</td>
    </tr>
    <tr>
      <th>22</th>
      <td>-0.798084</td>
      <td>-0.736176</td>
      <td>-0.786380</td>
      <td>0.009199</td>
      <td>-0.120425</td>
    </tr>
    <tr>
      <th>268</th>
      <td>2.282016</td>
      <td>2.053967</td>
      <td>1.495823</td>
      <td>0.009199</td>
      <td>0.216942</td>
    </tr>
    <tr>
      <th>371</th>
      <td>2.989460</td>
      <td>0.041490</td>
      <td>0.171371</td>
      <td>0.009199</td>
      <td>0.078830</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-897cea35-aa9f-45be-a6e7-def337264fce')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-897cea35-aa9f-45be-a6e7-def337264fce button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-897cea35-aa9f-45be-a6e7-def337264fce');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  



```python
df.drop('price', axis=1, inplace=False).columns
```

<pre>
Index(['LSTAT', 'LSTAT2', 'LSTAT3', 'LSTAT4', 'LSTAT5', 'LSTAT6', 'LSTAT7',
       'LSTAT8', 'LSTAT9', 'LSTAT10'],
      dtype='object')
</pre>

```python
coefficients = pd.DataFrame({'Features':df.drop('price', axis=1, inplace=False).columns})
coefficients['model_liner']=model_linear.coef_
coefficients['model_ridge']=model_ridge.coef_
coefficients['model_lasso']=model_lasso.coef_
coefficients['model_elastic_net']=model_net.coef_
coefficients
```


  <div id="df-9a008965-9421-492a-aa76-67ca5618f9cf">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Features</th>
      <th>model_liner</th>
      <th>model_ridge</th>
      <th>model_lasso</th>
      <th>model_elastic_net</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LSTAT</td>
      <td>10.485596</td>
      <td>-1.997427</td>
      <td>-0.0</td>
      <td>-0.154677</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LSTAT2</td>
      <td>-198.447363</td>
      <td>1.120503</td>
      <td>-0.0</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LSTAT3</td>
      <td>1218.536028</td>
      <td>0.705506</td>
      <td>-0.0</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LSTAT4</td>
      <td>-4039.520112</td>
      <td>-0.029523</td>
      <td>-0.0</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LSTAT5</td>
      <td>8029.887428</td>
      <td>-0.331986</td>
      <td>-0.0</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LSTAT6</td>
      <td>-9630.657680</td>
      <td>-0.305169</td>
      <td>-0.0</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>LSTAT7</td>
      <td>6399.310934</td>
      <td>-0.150135</td>
      <td>-0.0</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>LSTAT8</td>
      <td>-1554.634399</td>
      <td>0.007747</td>
      <td>-0.0</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>LSTAT9</td>
      <td>-523.920947</td>
      <td>0.115016</td>
      <td>-0.0</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>LSTAT10</td>
      <td>288.174050</td>
      <td>0.159595</td>
      <td>-0.0</td>
      <td>-0.000000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-9a008965-9421-492a-aa76-67ca5618f9cf')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-9a008965-9421-492a-aa76-67ca5618f9cf button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-9a008965-9421-492a-aa76-67ca5618f9cf');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  



```python

```


```python

```


```python

```


```python

```


```python

```

# Performance in the test set for 4 models.



```python
MSE_test = np.mean(np.square(df_pred['y_test'] - df_pred['y_hat_linear']))
RMSE_test = np.sqrt(MSE_test)
np.round(RMSE_test,3)
```

<pre>
0.541
</pre>

```python
MSE_test = np.mean(np.square(df_pred['y_test'] - df_pred['y_hat_ridge']))
RMSE_test = np.sqrt(MSE_test)
np.round(RMSE_test,3)
```

<pre>
0.559
</pre>

```python
MSE_test = np.mean(np.square(df_pred['y_test'] - df_pred['y_hat_lasso']))
RMSE_test = np.sqrt(MSE_test)
np.round(RMSE_test,3)
```

<pre>
1.006
</pre>

```python
MSE_test = np.mean(np.square(df_pred['y_test'] - df_pred['y_hat_net']))
RMSE_test = np.sqrt(MSE_test)
np.round(RMSE_test,3)
```

<pre>
0.898
</pre>

```python

```


```python

```


```python

```

# Plotting the regression coefficients vs alphas:


## 1) Ridge



```python
alpha_ridge = 10**np.linspace(-2,4,100)
```


```python
ridge = Ridge()
coefs_ridge = []

for i in alpha_ridge:
    ridge.set_params(alpha = i)
    ridge.fit(X_train, y_train)
    coefs_ridge.append(ridge.coef_)
    
np.shape(coefs_ridge)
```

<pre>
(100, 10)
</pre>

```python
plt.figure(figsize=(12,10))
ax = plt.gca()
ax.plot(alpha_ridge, coefs_ridge)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights: scaled coefficients')
plt.title('Ridge regression coefficients Vs. alpha')
plt.legend(df.drop('price',axis=1, inplace=False).columns)

plt.show()
```

<img src="data:image/png;base64,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"/>


```python

```

## 2) Lasso



```python
alpha_lasso = 10**np.linspace(-3,1,100)
```


```python
lasso = Lasso()
coefs_lasso = []

for i in alpha_lasso:
    lasso.set_params(alpha = i)
    lasso.fit(X_train, y_train)
    coefs_lasso.append(lasso.coef_)
    
np.shape(coefs_lasso)
```

<pre>
(100, 10)
</pre>

```python
plt.figure(figsize=(12,10))
ax = plt.gca()
ax.plot(alpha_lasso, coefs_lasso)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights: scaled coefficients')
plt.title('Lasso regression coefficients Vs. alpha')
plt.legend(df.drop('price',axis=1, inplace=False).columns)

plt.show()
```

<img src="data:image/png;base64,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"/>

## 3) ElasticNet



```python
alpha_elasticnet = 10**np.linspace(-3,2,100)
```


```python
elasticnet = ElasticNet()
coefs_elasticnet = []

for i in alpha_elasticnet:
    elasticnet.set_params(alpha = i)
    elasticnet.fit(X_train, y_train)
    coefs_elasticnet.append(elasticnet.coef_)
    
np.shape(coefs_elasticnet)
```

<pre>
(100, 10)
</pre>

```python
plt.figure(figsize=(12,10))
ax = plt.gca()
ax.plot(alpha_elasticnet, coefs_elasticnet)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights: scaled coefficients')
plt.title('Elastic Net regression coefficients Vs. alpha')
plt.legend(df.drop('price',axis=1, inplace=False).columns)

plt.show()
```

<img src="data:image/png;base64,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"/>

# Cross Validation


## 1) Ridge



```python
ridgecv = RidgeCV()
ridgecv.fit(X_train, y_train)
ridgecv.alpha_
```

<pre>
0.1
</pre>

```python
alpha_ridge_opt = ridgecv.alpha_
```

<pre>
0.1
</pre>
## 2) Lasso



```python
lassocv = LassoCV()
lassocv.fit(X_train, y_train)
lassocv.alpha_
```

<pre>
0.0007404280761639708
</pre>

```python
alpha_lasso_opt = lassocv.alpha_
```


```python
RMSE_CV=[]
iterator= np.arange(0.0,0.02,0.001)
for i in iterator:
    MSE = -cross_val_score(estimator = Lasso(alpha=i), X = X_train, y = y_train, cv = 5 , scoring="neg_mean_squared_error" )
    RMSE_CV.append(np.sqrt(MSE).mean())
    
output = pd.DataFrame(list(iterator), columns=['lambda_Lasso'])
output['RMSE_CV']=RMSE_CV

output.head()
```


  <div id="df-667b481c-bc17-4646-bf63-ae18e396e8a8">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lambda_Lasso</th>
      <th>RMSE_CV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000</td>
      <td>0.591622</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.001</td>
      <td>0.595970</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.002</td>
      <td>0.600641</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.003</td>
      <td>0.604491</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.004</td>
      <td>0.608323</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-667b481c-bc17-4646-bf63-ae18e396e8a8')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-667b481c-bc17-4646-bf63-ae18e396e8a8 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-667b481c-bc17-4646-bf63-ae18e396e8a8');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  



```python
output['RMSE_CV'].idxmin()
```

<pre>
0
</pre>

```python
output['lambda_Lasso'][output['RMSE_CV'].idxmin()]
```

<pre>
0.0
</pre>

```python
sns.lineplot(x='lambda_Lasso', y='RMSE_CV', data=output , color='r', label="RMSE_CV vs lambda_ridge")
plt.show()
```

<img src="data:image/png;base64,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"/>


```python

```

## 3) ElasticNet



```python
elasticnetcv = ElasticNetCV()
elasticnetcv.fit(X_train, y_train)
elasticnetcv.alpha_
```

<pre>
0.0014808561523279417
</pre>

```python
elasticnetcv.l1_ratio_
```

<pre>
0.5
</pre>

```python
from sklearn.model_selection import cross_val_score
import sklearn.metrics
```


```python
RMSE_CV=[]
iterator= np.arange(0.0,0.02,0.001)
for i in iterator:
    MSE = -cross_val_score(estimator = ElasticNet(alpha=i), X = X_train, y = y_train, cv = 5 , scoring="neg_mean_squared_error" )
    RMSE_CV.append(np.sqrt(MSE).mean())
    
output = pd.DataFrame(list(iterator), columns=['lambda_ElasticNet'])
output['RMSE_CV'] = RMSE_CV

output.head(25)
```


  <div id="df-bcf9a103-c004-45bf-99a3-44b98bc38892">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lambda_ElasticNet</th>
      <th>RMSE_CV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000</td>
      <td>0.591622</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.001</td>
      <td>0.597029</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.002</td>
      <td>0.602498</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.003</td>
      <td>0.606075</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.004</td>
      <td>0.609711</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.005</td>
      <td>0.612586</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.006</td>
      <td>0.614557</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.007</td>
      <td>0.616387</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.008</td>
      <td>0.617937</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.009</td>
      <td>0.619380</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.010</td>
      <td>0.620785</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.011</td>
      <td>0.622211</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.012</td>
      <td>0.623522</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.013</td>
      <td>0.624668</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.014</td>
      <td>0.625692</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.015</td>
      <td>0.626773</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.016</td>
      <td>0.627784</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.017</td>
      <td>0.628810</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.018</td>
      <td>0.629684</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.019</td>
      <td>0.630521</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-bcf9a103-c004-45bf-99a3-44b98bc38892')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-bcf9a103-c004-45bf-99a3-44b98bc38892 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-bcf9a103-c004-45bf-99a3-44b98bc38892');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  



```python
output['RMSE_CV'].idxmin()
```

<pre>
0
</pre>

```python
output['lambda_ElasticNet'][output['RMSE_CV'].idxmin()]
```

<pre>
0.0
</pre>

```python
sns.lineplot(x='lambda_ElasticNet', y='RMSE_CV', data=output , color='r', label="RMSE_CV vs lambda_ridge")
plt.show()
```

<img src="data:image/png;base64,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NmxEjW11HWvJksWMHn3W3rbhw0exYsVSAD755CNWr/6Wyy+fAoDfX0lKSu1/ZZrY4lRWUrp2DcUrvyFbV+271JWSSpujjqbN0X1oe3SfRu83aYhIXha7GpgmIrcD+cBUABGZC9yuqouAWcAQYBXuLcfzgae97X8uIiOBCtzblB9W1Xe8tunAcUDVrc1/VdW6/1xsov72t//HoYceznvvvcvdd/+Fvn3707595t52n8/HsGEj+Pvf/8bvf//nGvfRtWs3unbtxrhxE7jllhtYuPAjJk2ast9Zevc+glWrVtbanpCQwMiRZzJ37pskJiYyfPgoWrVqVeO6o0eP5Re/uIyf//xa3n13Pk888SwAp512Bn369OPLLz9nxozneOutN7j99jtrPWZOzg889NC/ePLJ5+nSpStff72cv/zlT3vbk5L2FeO4uDiSkloFvY7/STHaX0888Qjt22fyzDMvkJCQwE03XUt5eXmd2yQntw77tXDHcbj00stDOtMzsaF82zZKVn5N8cpvKFm9GqdsD8THkya9yZxwLm379KXVwT3wxTWNxxMjVlxUdTVuAai+fEzQzwHgZu9f9fVuqmPffuCa8CRtGoYNG8577y1g+vTn+NWvbvlR2/jx59C6dWuOP/7EHy0vKSnhm29WcMwxx+Hz+SgsLOSHH7Lp3LnrAWU499wL+NnPLuadd+YxcqR7+S0/fyeff/4pZ57pXrE866zx3HjjL0lISODuu2u/86pTp0707Hko999/Lz17HkqnTp0B2LJlM126dGXMmHF069adu+/+a52ZiouLSUhIJDMzk0AgwOuvv3pA7w3g66+Xs2nTJrp06cZbb73B4MHuWdPgwUOYN28uw4aNwO+vZMGCeRx0UCcAiooKOeywXiQkJLB+/VqWL1/GiBGj6zpMneo61sCBg5k//21OO204fr+f995bQIcObp/K0KGnMHv2LE455XTS0tIoLy/n++837r2Rw0RfoLycklUrKf7ma0pWfk1FrnshJ7FDFmnHn0DbPn1IPuIoOh3cMab7hGoTax36Zj9cffV1XHHFFC6++NIfLc/K6viTZS6H1157mfvu+ztJSa3w+/2MHHkmp56671LV/fffy5NPPrb39ZVX/oKhQ0+t8fgdOmTx8MP/5rHHHuTJJx8jObk1ycltmDJl37EPPfQwDjqoE+Xl5RxxxJF1vp8x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"/>


```python

```
