import robin_stocks as rh
import numpy as np

class metrics():
    """Class that contains functions that measure how stocks are
    doing"""

    def __init__(self, un, pw):
        """Initialization function for the metrics object. Just
        pass in the username (email) and password associated
        with the robin hood account you wish to use."""
        self.un = un
        self.pw = pw


    def stockdata_to_inputsymbols(stock_data):
        """Function to derive stock list from the stock data 
        that robin hood gives you using get_historical_data

        Parameters
            stock_data - Give stock data for any number of
                stocks in the form that you get by using the
                robin_stocks function get_historical_data

        Returns
            A list of stock symbols
        """

        '''Set up inputSymbols as a list and then populate it
        with the stock symbols from stock_data'''
        inputSymbols = []
        for i in np.arange(len(stock_data)):
            if stock_data[i]['symbol'] not in inputSymbols:
                inputSymbols.append(stock_data[i]['symbol'])
            else:
                continue

        return inputSymbols

    
    def dictionary_data(stock_data, data_point):
        """Function that takes in stock data in the form that
        robin hood gives it to you from the get_historical_data
        function and turns it into a dictionary of the form
        {stock symbol: data_list} where data_list is a numpy array
        containing all the data points of the wanted category.

        Parameters
            stock_data - Give data for any number of stocks in
                the form that the get_historical_data function
                in robin_stocks.py gives it to you
            data_point - Give the name of the data metric that
                you are interested in. Options are open_price,
                close_price, high_price, low_price, or volume.
                Give this as a string

        Returns
            A dictionary that is of the form
            {stock symbol: data list}
        """
        
        '''Use the stock_data to get a list of stocks that we are
        interested in'''
        inputSymbols = self.stockdata_to_inputsymbols(stock_data)

        '''Set up dictionary for the given data point and then
        populate it with the wanted data'''
        num_points = int(len(stock_data)/len(inputSymbols))
        dp_dic = {}
        j=0
        for i in np.arange(len(inputSymbols)):
            data = []
            k=0
            while k < num_points:
                data.append(float(stock_data[j]['{}'.format(data_point)]))
                j+=1
                k+=1
            dp_dic['{}'.format(inputSymbols[i])] = data

        '''Return dp_dic'''
        return dp_dic


        
        
    def bollinger_bands(self, inputSymbols, interval='day', span='year'):
        """Function that calculates bollinger bands for a given
        list of stocks

        Parameters
            inputSymbols - Give the list of symbols
                to calculated bollinger bands for (might be a list
                with just one item)
            timelength - Give the length of time you want the
                bollinger bands to be calculated over. Default
                is one day
            span - Give the span of time which the averages and 
                standard deviations should be calculated over.
                Default is one year. Options are week, month,
                3month, year, or 5year

        Returns
            A dictionary with the symbols matched with its
            bollinger band values in the following form
            (mean - 2 standard dev., mean - st dv, ..., mean + 2 st dv)
        """

        '''login to robin hood and get stock data for given stocks, 
        interval, and span'''
        rh.login(self.un, self.pw)
        stock_data = rh.stocks.get_stock_historicals(inputSymbols=inputSymbols, interval=interval, span=span)

        '''use the dictionary_data function above to put the data
        into a dictionary so it's easier to use'''
        high_dic = self.dictionary_data(stock_data, data_point='high')

        '''calculate the bollinger bands. Five numbers for each
        entry represent the mean and then one and two standard
        deviations either up or down from the mean'''
        bb_dic = {}
        for i in np.arange(len(inputSymbols)):
            highs = high_dic['{}'.format(inputSymbols[i])]
            mean = np.mean(highs)
            std = np.std(highs)
            oneup = mean + std
            onedown = mean - std
            twoup = mean + 2*std
            twodown = mean - 2*std
            bbands = [twodown, onedown, mean, oneup, twoup]
            bb_dic['{}'.format(inputSymbols[i])] = np.array(bbands)

        '''return the bollinger bands dictionary'''
        return bb_dic



    def moving_average(self, inputSymbols, span='year'):
        """Function that computes a moving average of stock prices
        (daily high or low) over a given time period.

        Parameters
            inputSymbols - Give the input symbols for the stocks
                which you want a moving average for. Even if it's
                just one stock, enter it as a list
            span - Give the length of time over which the moving
                average should be calculated. Either give
                day, week, month, 3month, year, or 5year

        Returns
        """
        
        '''login to robin hood and get stock data for given stocks,
        interval, and span'''
        rh.login(self.un, self.pw)
        stock_data = rh.stocks.get_stock_historicals(inputSymbols=inputSymbols, interval='day', span=span)

        '''Take the data and put it into a dictionary so it's easy
        to use. Use the dictionary_data function to do this. Also use
        stockdata_to_inputsymbols to get a list of stocks that we are
        working with'''
        high_dic = self.dictionary_data(stock_data, data_point='high')
        inputSymbols = self.stockdata_to_inputsymbols(stock_data)

        '''For each stock, get the high data and then take the average.
        The span of the average is decide when we pull the data from
        robin hood to get stock_data'''
        mean_dic = {}
        for i in np.arange(len(inputSymbols)):
            data = high_dic['{}'.format(inputSymbols[i])]
            mean = np.mean(data)
            mean_dic['{}'.format(inputSymbols[i])] = mean

        '''return dictionary containing moving averages'''
        return mean_dic
