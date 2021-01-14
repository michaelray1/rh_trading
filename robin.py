import robin_stocks as rh
import numpy as np

class statistics():
    """Class that calculates different statistics for stocks.
    Takes in inputSymbols which is a list of strings with
    the stock tickers as elements"""

    def __init__(self, un, pw, inputSymbols):
        """Initialization function for the metrics object. Just
        pass in the username (email) and password associated
        with the robin hood account you wish to use. Also give
        a list of strings for inputSymbols. Each string should
        be a stock ticker."""
        self.un = un
        self.pw = pw
        self.inputSymbols = inputSymbols

        
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

        '''Set up dictionary for the given data point and then
        populate it with the wanted data'''
        num_points = int(len(stock_data)/len(self.inputSymbols))
        dp_dic = {}
        j=0
        for i in np.arange(len(self.inputSymbols)):
            data = []
            k=0
            while k < num_points:
                data.append(float(stock_data[j]['{}'.format(data_point)]))
                j+=1
                k+=1
            dp_dic['{}'.format(self.inputSymbols[i])] = data

        '''Return dp_dic'''
        return dp_dic


        
        
    def bollinger_bands(self, interval='day', span='year'):
        """Function that calculates bollinger bands for a given
        list of stocks

        Parameters
            inputSymbols - Give the list of symbols
                to calculated bollinger bands for (might be a list
                with just one item)
            interval - Give the length of time you want the
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
        stock_data = rh.stocks.get_stock_historicals(inputSymbols=self.inputSymbols, interval=interval, span=span)

        '''use the dictionary_data function above to put the data
        into a dictionary so it's easier to use'''
        high_dic = self.dictionary_data(stock_data, data_point='high')

        '''calculate the bollinger bands. Five numbers for each
        entry represent the mean and then one and two standard
        deviations either up or down from the mean'''
        bb_dic = {}
        for i in np.arange(len(self.inputSymbols)):
            highs = high_dic['{}'.format(self.inputSymbols[i])]
            mean = np.mean(highs)
            std = np.std(highs)
            oneup = mean + std
            onedown = mean - std
            twoup = mean + 2*std
            twodown = mean - 2*std
            bbands = [twodown, onedown, mean, oneup, twoup]
            bb_dic['{}'.format(self.inputSymbols[i])] = np.array(bbands)

        '''return the bollinger bands dictionary'''
        return bb_dic



    def moving_average(self, span='year'):
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
        stock_data = rh.stocks.get_stock_historicals(inputSymbols=self.inputSymbols, interval='day', span=span)

        '''Take the data and put it into a dictionary so it's easy
        to use. Use the dictionary_data function to do this. Also use
        stockdata_to_inputsymbols to get a list of stocks that we are
        working with'''
        high_dic = self.dictionary_data(stock_data, data_point='high')

        '''For each stock, get the high data and then take the average.
        The span of the average is decide when we pull the data from
        robin hood to get stock_data'''
        mean_dic = {}
        for i in np.arange(len(self.inputSymbols)):
            data = high_dic['{}'.format(self.inputSymbols[i])]
            mean = np.mean(data)
            mean_dic['{}'.format(self.inputSymbols[i])] = mean

        '''return dictionary containing moving averages'''
        return mean_dic




Class metrics():
    """Class whose job is to get True or False values for a
    list of stocks which are not currently in your portfolio. 
    True indicates that we should buy the stock because we
    think it will go up. False indicates that we should not buy.
    Requires that you give it a statistics object (defined above)
    """

    def __init__(self, statistics):
        self.statistics = statistics

    def bbands(self, num_bands=0, interval='day', span='year'):
        """Set metrics.bbands = True if the stock is within 
        num_bands of the bottom band.
        Sets metrics.bbands = False if the stock is not 
        within num_bands of the bottom band
        
        Parameters
        num_bands -  must be an integer between 0 and 4. 
            0 means a buy flag is produced only if the stock 
            is below 2 standard deviations lower than its mean. 
            4 means a buy flag is produced only if the stock 
            is below 2 standard deviations above its mean.
            Default value is 0.
        interval - Give the length of time you want the
            bollinger bands to be calculated over. Default
            is one day
        span - Give the span of time which the averages and                                                    
            standard deviations should be calculated over as a string.
            Default is one year. Options are week, month
            3month, year, or 5year

        Returns
        A dictionary where the keys are the inputSymbols
        or the stock tickers associated with the statistics object.
        The values in the dictionary are either 1 or 0. 1 indicates
        a buy flag is associated with that stock. 0 indicates there
        is no buy flag for that stock.
        """

        bollinger = self.statistics.bollinger_bands()

        '''Evaluate whether the stocks are currently within
        num_bands of the bottom band.'''
        current_prices = rh.stocks.get_latest_price(inputSymbols=self.statistics.inputSymbols)
        boolean_dic = {}
        for i in np.arange(len(self.statistics.inputSymbols)):
            if float(current_prices[i]) <= bollinger[self.statistics.inputSymbols[i]][num_bands]:
                boolean_dic['{}'.format(self.statistics.inputSymbols[i])] = 1.0
            else:
                boolean_dic['{}'.format(self.statistics.inputSymbols[i]] = 0.0

        return boolean_dic
