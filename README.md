# Stock Price Forecasting

This is an attempt to forecast the closing price of a stock given its previous closing prices.

This repository contains:

1. The [python code](Code.py)
2. The generated [prediction graph](Forecast.png)
3. The [sample dataset](Sample_dataset_TCS-NS.csv)
3. [ReadMe file](README.md) itself


## Table of Contents

- [About](#about)
- [To Run](#to-run)


## About

### Dataset
The dataset used in this repo is the historical data of [Tata Consultancy Services](https://finance.yahoo.com/quote/TCS.NS/history?period1=1029110400&period2=1637971200&interval=1d&frequency=1d&filter=history) (a company listed in National Stock Exchange, India) from August 12, 2002 to November 26, 2021. The dataset was scraped from [Yahoo Finance](https://finance.yahoo.com) using Selenium. The data from Selenium is parsed using BeautifulSoup and converted to their respective datatypes before storing it in the Pandas dataframe.

### Model
A stacked 2 layered-LSTM model followed by a Dense layer is employed. The inputs from the training dataset are concatenated with their true values whereas the inputs from the test dataset is concatenated with all zeroes. The model is expected to predict the true values of the test dataset. Shown below is a stacked LSTM model.

The hyper-parameters can be changed in the [python file](Code.py).


## To Run

Make sure you have the following libraries installed to run the code.
```
numpy
pandas
matplotlib
bs4
selenium
scikit-learn
plotly
keras
```

Download the following files in the same directory.
```
Code.py 
```

Run Code.py 
```
cd directory-where-the-files-are-saved
python3 Code.py
```
