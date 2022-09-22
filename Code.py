"""IN-BUILT MODULES"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import time

from datetime import date, datetime, timezone
from dateutil.relativedelta import relativedelta
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping



class Stock:
    sector, industry, company, logo, currency, historical_data = '', '', '', '', '', ''
    
    def __init__(self, ticker, chromedriver_path = None):
        caps = DesiredCapabilities.CHROME
        caps["goog:loggingPrefs"] = {"performance": "ALL"}

        options = Options()
        options.add_argument("--headless")
        options.add_argument("--start-maximized")
        
        if chromedriver_path is None:
            chromedriver_path = 'C:\\Users\\vikne\\AppData\\Local\\Programs\\Python\\Python39\\Lib\\site-packages\\chromedriver.exe'
        
        
        driver = webdriver.Chrome(chromedriver_path, desired_capabilities = caps, options = options)

#####################################################################################################################
        """ PART 1 """

        URL = "https://finance.yahoo.com/quote/{}/profile?p={}".format(ticker, ticker)
        driver.get(URL)
        time.sleep(2)

        webpage = driver.page_source
        HTMLPage = BeautifulSoup(webpage, 'html.parser')

        Paragraph = HTMLPage.find('p', class_= 'D(ib) Va(t)')
        Sector_Industry = Paragraph.find_all('span')
        self.sector = Sector_Industry[1].text
        self.industry = Sector_Industry[3].text

        self.company = HTMLPage.find('h3', class_= 'Fz(m) Mb(10px)').text

        Paragraph = HTMLPage.find('p', class_= 'D(ib) W(47.727%) Pend(40px)')
        Logo = Paragraph.find_all('a')
        self.logo = 'https://logo.clearbit.com/' + Logo[1].text.split('://')[1].replace('www.', '')

        self.currency = HTMLPage.find('div', class_= 'C($tertiaryColor) Fz(12px)').text[-3 : ]
        
#######################################################################################################################
        """ PART 2 """

        start = int(datetime.strptime('1970-01-01', "%Y-%m-%d").replace(tzinfo = timezone.utc).timestamp())
        end = int(datetime.strptime(date.today().strftime('%Y-%m-%d'), "%Y-%m-%d").replace(tzinfo = timezone.utc).timestamp())
        URL = "https://finance.yahoo.com/quote/{}/history?period1={}&period2={}&interval=1d&frequency=1d&filter=history".format(ticker, start, end)

        driver.get(URL)
        time.sleep(2)

        scroll = (end - start) // (24 * 60 * 60 * 100)
        for i in range(0, scroll + 1):
            driver.execute_script("window.scrollBy(0,5000)")
        time.sleep(2)

        webpage = driver.page_source
        HTMLPage = BeautifulSoup(webpage, 'html.parser')
        Table = HTMLPage.find('table', class_='W(100%) M(0)')
        Rows = Table.find_all('tr', class_='BdT Bdc($seperatorColor) Ta(end) Fz(s) Whs(nw)')
        
        data = []
        for i in range(0, len(Rows)):
            RowDict = {}
            Values = Rows[i].find_all('td')
            
            if len(Values) == 7:
                try:
                    RowDict["Date"] = Values[0].find('span').text.replace(',', '')
                    RowDict["Open"] = float(Values[1].find('span').text.replace(',', ''))
                    RowDict["High"] = float(Values[2].find('span').text.replace(',', ''))
                    RowDict["Low"] = float(Values[3].find('span').text.replace(',', ''))
                    RowDict["Close"] = float(Values[4].find('span').text.replace(',', ''))
                    RowDict["Adj Close"] = float(Values[5].find('span').text.replace(',', ''))
                    RowDict["Volume"] = int(Values[6].find('span').text.replace(',', ''))
                
                except:
                    pass
                
                else:
                   data.append(RowDict)
                
                finally:
                    i = i + 1
    
        self.historical_data = pd.DataFrame(data)
        self.historical_data['Date'] = pd.to_datetime(self.historical_data.Date)
        self.historical_data = self.historical_data.sort_values(by = 'Date')
        
        driver.close()
    
    def get_data(self):
        return self.sector, self.industry, self.company, self.logo, self.currency, self.historical_data



class PredictStockPrice:
    def __init__(self, data, end_date):
        self.df = data.loc[(data['Date'] >= '1970-01-01') & (data['Date'] <= end_date)]
        self.data_size = int(self.df.shape[0] * 0.9)
        self.input_size = 60
        self.last_date = end_date

        self.df['Date'] = pd.to_datetime(self.df.Date, format = '%d-%m-%Y')
        self.df.index = self.df['Date']
    
    
    def extract_data(self, columns):        
        data = self.df.sort_index(ascending = True, axis = 0)
        new_data = pd.DataFrame(index = range(0, len(data)), columns = columns)
        for i in range(0, len(data)):
            for col in columns:
                new_data[col][i] = data[col][i]
        
        new_data.index = new_data.Date
        new_data.drop('Date', axis = 1, inplace = True)
        
        return new_data
    

    def normalize_data(self, dataset, train_data, test_data):
        self.scaler = MinMaxScaler(feature_range = (0, 1))
        self.scaled_data = self.scaler.fit_transform(dataset)

        self.X_train_data, self.Y_train_data = [], []

        for i in range(self.input_size, len(train_data)):
            self.X_train_data.append(self.scaled_data[i - self.input_size : i, 0])
            self.Y_train_data.append(self.scaled_data[i, 0])
        
        self.X_train_data, self.Y_train_data = np.array(self.X_train_data), np.array(self.Y_train_data)
        self.X_train_data = np.reshape(self.X_train_data, (self.X_train_data.shape[0], self.X_train_data.shape[1], 1))

        self.input = dataset[len(dataset) - len(test_data) - self.input_size : ]
        self.input = self.input.reshape(-1, 1)
        self.input = self.scaler.transform(self.input)

        self.X_test_data = []

        for i in range(self.input_size, self.input.shape[0]):
            self.X_test_data.append(self.input[i - self.input_size : i, 0])
        
        self.X_test_data = np.array(self.X_test_data)
        self.X_test_data = np.reshape(self.X_test_data, (self.X_test_data.shape[0], self.X_test_data.shape[1], 1))
        
    
    def build_train_LSTM_model(self):
        self.lstm_model = Sequential()
        self.lstm_model.add(LSTM(units = 50, return_sequences = True, input_shape = (self.X_train_data.shape[1], 1)))
        self.lstm_model.add(LSTM(units = 50))
        self.lstm_model.add(Dense(1))

        self.lstm_model.compile(loss = 'mean_squared_error', optimizer = 'adam')
        
        es = EarlyStopping(monitor = 'loss', mode = 'min', verbose = 1, patience = 5)

        history = self.lstm_model.fit(self.X_train_data, self.Y_train_data, epochs = 2, batch_size = 1, verbose = 2, callbacks = [es], validation_split = 0.1)
        
        return history


    def predict_price(self):
        predicted_price = self.lstm_model.predict(self.X_test_data)
        predicted_price = self.scaler.inverse_transform(predicted_price)

        return predicted_price

    def get_candlesticks(self):
        close = self.extract_data(columns = ['Date', 'Close'])
        dataset = close.values
        size = int(self.df.shape[0] * 0.9)
        train, test = dataset[0 : size, : ], dataset[size : , : ]
        self.normalize_data(dataset = dataset, train_data = train, test_data = test)
        history = self.build_train_LSTM_model()
        closing_price = self.predict_price()

        predictions = close[size : ]
        predictions.insert(len(predictions.columns), 'Predictions', closing_price)

        rmse = np.sqrt(np.mean(np.power((test - closing_price),2)))
    
        return close[ : size], predictions, rmse, history



all_options = {
    'USA': ['NASDAQ', 'NYSE'],
    'INDIA': ['NSE', 'BSE']
}

extensions = {
    'NASDAQ' : '',
    'NYSE' : '',
    'NSE' : '.NS',
    'BSE' : '.BO'
}

start_date = datetime.strptime(datetime.strftime(date.today() - relativedelta(years = +1), '%Y-%m-%d'), '%Y-%m-%d')
end_date = datetime.strptime(datetime.strftime(date.today(), '%Y-%m-%d'), '%Y-%m-%d')


print('Select Country', list(all_options.keys()), ': ', end = '')
country = input().upper()

if country not in all_options.keys():
    print('Country does not exist.....Try again')
    exit()

print('Select Exchange', list(all_options[country]), ': ', end = '')
exchange = input().upper()

if exchange not in all_options[country]:
    print('{} Stock Exchange does not exist in {}.....Try again'.format(exchange, country))
    exit()

print('Enter Stock Ticker : ', end = '')
ticker = input().upper()


ticker_symbol = ticker + extensions[exchange]
company_info = Stock(ticker = ticker_symbol)
sector, industry, company_name, logo, currency, df = company_info.get_data()
df.drop(['Volume', 'Adj Close'], axis = 1, inplace = True)

data = df.loc[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
data['Date'] = data['Date'].dt.strftime('%b %d, %Y')

fig = go.Figure(data = [go.Candlestick(x = data['Date'], open = data['Open'], high = data['High'], low = data['Low'], close = data['Close'])])
fig.update_layout(title = "Stock Price vs Date", yaxis_title = "Price (in {})".format(currency))
fig.show()

stock = PredictStockPrice(data = df, end_date = end_date)
train_data, prediction, rmse, history = stock.get_candlesticks()

print("\n\nCompany Name : ", company_name)
print("Sector : ", sector)
print("Industry : ", industry)

print('\n\nRMSE Error : ', rmse, end = '\n\n')

column = 'Close'


plt.plot(train_data, label = 'Train Data', color = 'blue')
plt.plot(prediction[['Close']], label = 'Actual Price', color = 'green')
plt.plot(prediction[['Predictions']], label = 'Forecasted Price', color = 'orange')
plt.legend()
plt.show()

plt.plot(prediction[['Close']], label = 'Actual Price', color = 'green')
plt.plot(prediction[['Predictions']], label = 'Forecasted Price', color = 'orange')
plt.legend()
plt.show()

plt.plot(range(1, len(history.history['loss']) + 1), history.history['loss'], color = 'red', label = 'Train Loss')
plt.plot(range(1, len(history.history['val_loss']) + 1), history.history['val_loss'], color = 'blue', label = 'Test Loss')
plt.legend()
plt.show()