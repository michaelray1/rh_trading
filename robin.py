import numpy as np
import robin_stocks.robinhood as rh
import tensorflow as tf
from tensorflow.keras.activations import relu



class statistics():
    """Class that calculates different statistics for stocks.
    Takes in inputSymbols which is a list of strings with
    the stock tickers as elements"""

    def __init__(self):
        """Initialization function for the statistics object."""
        pass


    def sd_to_is(self, stock_data):
        """Converts stock data in the form that the robin stocks
        function get_stock_historicals gives it to you to 
        just a list of the stock tickers associated with that data.

        Parameters
        stock_data - Give stock data just as it is given from the
            robin_stocks function get_stock_historicals

        Returns
        A list of strings that are all the stock tickers present
        in the stock_data
        """
        inputSymbols = []
        for i in np.arange(len(stock_data)):
            if stock_data[i]['symbol'] in inputSymbols:
                pass
            else:
                inputSymbols.append(stock_data[i]['symbol'])

        return inputSymbols
        
        
    def dictionary_data(self, stock_data, data_point):
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
        populate it with the wanted data. Data will go in order
        from past to present ending with the previous days data.'''
        inputSymbols = self.sd_to_is(stock_data)
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
        
        
    def bollinger_bands(self, stock_data):
        """Function that calculates bollinger bands for a given
        list of stocks

        Parameters
            stock_data - Give a dictionary of the same form that
                you would get from the function get_stock_historicals

        Returns
            A dictionary with the symbols matched with its
            bollinger band values in the following form
            (mean - 2 standard dev., mean - st dv, ..., mean + 2 st dv)
        """

        '''use the dictionary_data function above to put the data
        into a dictionary so it's easier to use'''
        close_dic = self.dictionary_data(stock_data=stock_data, data_point='close_price')
        inputSymbols = self.sd_to_is(stock_data=stock_data)

        '''calculate the bollinger bands. Five numbers for each
        entry represent the mean and then one and two standard
        deviations either up or down from the mean'''
        bb_dic = {}
        for i in np.arange(len(inputSymbols)):
            prices = close_dic['{}'.format(inputSymbols[i])]
            mean = np.mean(prices)
            std = np.std(prices)
            oneup = mean + std
            onedown = mean - std
            twoup = mean + 2*std
            twodown = mean - 2*std
            bbands = [twodown, onedown, mean, oneup, twoup]
            bb_dic['{}'.format(inputSymbols[i])] = np.array(bbands)

        '''return the bollinger bands dictionary'''
        return bb_dic


    def moving_average(self, stock_data):
        """Function that computes a moving average of stock prices
        (daily closing price) over a given time period.

        Parameters
            stock_data - Give a dictionary of the same form that
                you would get from the function get_stock_historicals

        Returns
        A dictionary where the keys are stock tickers and the values
        are the moving average of each stock over the span given.
        """

        '''Take the data and put it into a dictionary so it's easy
        to use. Use the dictionary_data function to do this.'''
        close_dic = self.dictionary_data(stock_data, data_point='close_price')
        inputSymbols = self.sd_to_is(stock_data=stock_data)

        '''For each stock, get the close data and then take the average.
        The span of the average is decide when we pull the data from
        robin hood to get stock_data'''
        mean_dic = {}
        for i in np.arange(len(inputSymbols)):
            data = close_dic['{}'.format(inputSymbols[i])]
            mean = np.mean(data)
            mean_dic['{}'.format(inputSymbols[i])] = mean

        '''return dictionary containing moving averages'''
        return mean_dic


    def exp_average(self, data, sigma=7):
        """Function that computes an exponential average 
        of data. It takes data, then multiplies by one half of a
        gaussian curve, then divides by the integral of the
        gaussian. This results in a weighted average.

        Parameters
            data - Give a numpy array of any length (but dimension 1)
                that represents the data you want to compute a
                weighted average for
            sigma - Give an integer that represents the standard
                deviation of your gaussian. Default is 7 (days)

        Returns
        A numpy array of the same length as data. It is a weighted
        average of the data with a gaussian applied as the weights.
        """
        
        '''We want x to be the first half of the domain for the gaussian
        distribution and to be as long as our data is'''
        x = np.arange(start=-len(data)+1, stop=1, step=1)
        gaussian = np.e**(-(x**2/(2*sigma**2)))
        smooth_data = data*gaussian
        mean = np.sum(smooth_data)/np.sum(gaussian)

        '''return mean_dic'''
        return mean
            
    
    def rsi(self, stock_data, sigma=7):
        """Function that calculates the relative strength index for
        the inputSymbols associated with the statistics object. This
        calculates the moving average for the rsi calculation over
        
        Parameters
            stock_data - Give a dictionary of the same form that
                you would get from the function get_stock_historicals
            sigma - Give an integer that is the sigma you want
                when constructing the gaussian function which
                smooths the stock data

        Returns
        A dictionary whose keys are the stock tickers and whose
        values are the relative strength index of the stock
        """

        '''Take the data and put it into a dictionary so it's easy
        to use. Use the dictionary_data function to do this.'''
        close_dic = self.dictionary_data(stock_data=stock_data, data_point='close_price')
        inputSymbols = self.sd_to_is(stock_data=stock_data)

        '''Create a dictionary for rsi values. This will be filled
        in the loop below and returned at the end of the function.'''
        rsi_dic = {}

        '''loop over inputSymbols (stocks)'''
        for i in np.arange(len(inputSymbols)):
            
            '''initialize up and down arrays for rsi calculation'''
            up_array = []
            down_array = []

            '''loop over the length of the stock data'''
            for j in np.arange(len(close_dic[inputSymbols[i]])):

                '''append the difference in price to up array if the 
                price went up. append to down array if it went down.
                if the price stayed the same, append a zero to both.
                '''
                if close_dic[inputSymbols[i]][j] > close_dic[inputSymbols[i]][j-1]:
                    up_array.append(np.abs(close_dic[inputSymbols[i]][j] - close_dic[inputSymbols[i]][j-1]))
                    down_array.append(0)
                elif close_dic[inputSymbols[i]][j] < close_dic[inputSymbols[i]][j-1]:
                    up_array.append(0)
                    down_array.append(np.abs(close_dic[inputSymbols[i]][j] - close_dic[inputSymbols[i]][j-1]))
                else:
                    up_array.append(0)
                    down_array.append(0)

            up_array = np.array(up_array)
            down_array = np.array(down_array)

            '''compute the exponentially weighted average of up 
            and down arrays'''
            up_exp_avg = self.exp_average(data=up_array, sigma=7)
            down_exp_avg = self.exp_average(data=down_array, sigma=7)

            '''compute the relative strength and relative strength
            index. Then place the rsi in the rsi dictionary'''
            rel_strength = up_exp_avg/down_exp_avg
            rel_strength_index = 100 - (100/(1 + rel_strength))

            rsi_dic[inputSymbols[i]] = rel_strength_index

        return rsi_dic


    



        
class metrics():
    """Class whose job is to get True or False values for a
    list of stocks which are not currently in your portfolio. 
    True indicates that we should buy the stock because we
    think it will go up. False indicates that we should not buy.
    Requires that you give it a statistics object (defined above)
    """

    def __init__(self, statistics):
        self.statistics = statistics

        
    def bbands_bottom(self, stock_data, num_bands=1):
        """Function that compares bollinger bands on a collection
        of stocks to their current price. Returns a dictionary
        where the keys are stock tickers and the values are
        1 if we should buy based on bollinger band assessment,
        or zero if we shouldn't buy based on bollinger band
        assessment. This assessment is made based on whatever
        the last price in stock_data is. This provides functionality
        for backtesting.

        Parameters
        stock_data - Give a dictionary of the same form that       
            you would get from the function get_stock_historicals
        num_bands -  must be an integer between 0 and 4. 
            0 means a buy flag is produced only if the stock 
            is below 2 standard deviations lower than its mean. 
            4 means a buy flag is produced only if the stock 
            is below 2 standard deviations above its mean.
            Default value is 0.

        Returns
        A dictionary where the keys are the inputSymbols
        or the stock tickers associated with the statistics object.
        The values in the dictionary are either 1 or 0. 1 indicates
        a buy flag is associated with that stock. 0 indicates there
        is no buy flag for that stock.
        """

        bollinger = self.statistics.bollinger_bands(stock_data=stock_data)
        inputSymbols = self.statistics.sd_to_is(stock_data=stock_data)

        '''Split the data into separate data sets for each stock, 
        then populate a dictionary full of the latest prices on each
        stock within the data set.'''
        closing_prices = self.statistics.dictionary_data(stock_data=stock_data, data_point='close_price')
        last_prices = {}
        for i in np.arange(len(inputSymbols)):
            last_prices['{}'.format(inputSymbols[i])] = closing_prices['{}'.format(inputSymbols[i])][-1]
        
        '''Evaluate whether the stocks are currently within
        num_bands of the bottom band.'''
        boolean_dic = {}
        for i in np.arange(len(inputSymbols)):
            if float(last_prices['{}'.format(inputSymbols[i])]) <= bollinger['{}'.format(inputSymbols[i])][num_bands]:
                boolean_dic['{}'.format(inputSymbols[i])] = 1.0
            else:
                boolean_dic['{}'.format(inputSymbols[i])] = -1.0

        return boolean_dic


    def ma_crossover(self, stock_data, short_period=25, long_period=250):
        """Function that does an assessment based on long-term
        vs short-term moving averages. Returns a dictionary where
        the keys are stock tickers and the values are 1 if the
        short_period moving average is above the long_period
        moving average.

        Parameters
        stock_data - Give a dictionary of the same form that
            you would get from the function get_stock_historicals
        short_period - Give an integer which represents the short
            time period moving average you want to calculate in days.
        long_period - Give an integer which represents the long
            time period moving average you want to calculate in days

        Returns
        A dictionary where the keys are the stock tickers associated
        with the statistics object and the values are 1 if we should
        buy or 0 if we shouldn't. We buy when the short_period 
        moving average is above the long_period moving average.
        """
        inputSymbols = self.statistics.sd_to_is(stock_data=stock_data)
        points_per_stock = int(len(stock_data)/len(inputSymbols))

        stock_data_short = []
        stock_data_long = []
        for j in np.arange(len(inputSymbols)):
            stock_data_short.append(stock_data[(j+1)*points_per_stock-short_period: (j+1)*points_per_stock])
            stock_data_long.append(stock_data[(j+1)*points_per_stock-long_period: (j+1)*points_per_stock])

        '''Reshape stock_data_long and stock_data_short so that they
        have the same structure as if we had called rh.get_stock_historicals
        with the time limit already enforced.'''
        stock_data_short = np.array(stock_data_short)
        stock_data_long = np.array(stock_data_long)
        stock_data_short = stock_data_short[0,:]
        stock_data_long = stock_data_long[0,:]

        short_dic = self.statistics.moving_average(stock_data=stock_data_short)
        long_dic = self.statistics.moving_average(stock_data=stock_data_long)

        boolean_dic = {}
        for i in np.arange(len(inputSymbols)):
            if short_dic['{}'.format(inputSymbols[i])] > long_dic['{}'.format(inputSymbols[i])]:
                boolean_dic['{}'.format(inputSymbols[i])] = 1.0
            else:
                boolean_dic['{}'.format(inputSymbols[i])] = -1.0

        return boolean_dic


    def rsi_crossover(self, stock_data, rsi_cutoff=30, strategy='below'):
        """Function that does an assessment based on relative
        strength index. Returns a dictionary where the keys are
        stock tickers and the values are 1 if the rsi value of 
        a stock is either above or below (based on value of
        strategy provided to the function) rsi_cutoff and a zero
        if not.

        Parameters
        stock_data - Give a dictionary of the same form that
            you would get from the function get_stock_historicals
        rsi_cutoff - Give a cutoff value for the relative strength
            index. This should be a real number between 0 and 100.
        strategy - Give a string that is either 'below' or 'above'.
            This tells the function to generate a buy signal when
            the stock goes either above or below the rsi_cutoff.

        Returns
        A dictionary where the keys are stock tickers and the values
        are either 1 or 0 based on if the stock is above or below 
        (based on value of strategy given to the function) the
        rsi_cutoff value.
        """

        rsi_dic = self.statistics.rsi(stock_data=stock_data, sigma=7)
        inputSymbols = self.statistics.sd_to_is(stock_data=stock_data)

        boolean_dic = {}
        
        '''loop over the stocks in inputSymbols'''
        for i in np.arange(len(inputSymbols)):
            '''if strategy is below, then return a buy signal if
            the rsi is below rsi_cutoff'''
            if strategy == 'below':
                if float(rsi_dic['{}'.format(inputSymbols[i])]) < rsi_cutoff:
                    boolean_dic['{}'.format(inputSymbols[i])] = 1.0
                else:
                    boolean_dic['{}'.format(inputSymbols[i])] =	-1.0

                    '''if strategy is above, then return a buy signal
                    if the rsi is above rsi_cutoff'''
            elif strategy == 'above':
                if float(rsi_dic['{}'.format(inputSymbols[i])]) < rsi_cutoff:
                    boolean_dic['{}'.format(inputSymbols[i])] =	-1.0
                else:
                    boolean_dic['{}'.format(inputSymbols[i])] =	1.0

                    '''if strategy is neither above nor below, throw a
                    ValueError'''
            else:
                raise ValueError('strategy must be either above or below. Your strategy was {}'.format(strategy))


        return boolean_dic


    def volume_crossover(self, stock_data, strategy='above', time=100):
        """
        function which takes in stock data and a strategy
        and outputs a dictionary whose keys are stock tickers
        and the values are (assuming strategy is above and time=100) 
        0 if volume is below the 100 day mean and 1 if the volume
        is above the 100 day mean.

        Parameters
        stock_data - Give a dictionary of the same form that        
            you would get from the function get_stock_historicals
        strategy - Give a string that is either 'above' or 'below'.
            'above' indicates you want to send a buy signal when
            the volume is above average, 'below' indicates the opposite
        time - Give an integer that is the number of days you want to
            calculate the mean volume over. Default is 100.

        Returns
        A dictionary whose keys are stock tickers in stock_data and whose
        values are 1's and 0's depending on if the volume is higher or 
        lower than the mean volume.
        """

        '''derive inputSymbols and stock volume dictionary from the stock_data'''
        inputSymbols = self.statistics.sd_to_is(stock_data=stock_data)

        points_per_stock = int(len(stock_data)/len(inputSymbols))

        boolean_dic = {}

        '''loop over stock tickers'''
        timed_stock_data = []
        for j in np.arange(len(inputSymbols)):
            '''cut the data off so that we calculate the average of the
            volume over the correct time interval'''
            timed_stock_data.append(stock_data[(j+1)*points_per_stock-time: (j+1)*points_per_stock])

        '''Reshape timed_stock_data so that it matches the data structure
        as if we had called rh.get_stock_historicals with the time period
        enforced originally'''
        timed_stock_data = np.array(timed_stock_data)
        timed_stock_data = timed_stock_data[0,:]
            
        stock_volume_dictionary = self.statistics.dictionary_data(stock_data=timed_stock_data, data_point='volume')

        for i in np.arange(len(inputSymbols)):
            volumes = stock_volume_dictionary['{}'.format(inputSymbols[i])]
            mean_volume = np.mean(volumes)
            current_volume = volumes[-1]

            '''Here's the condition for buy/not buy signal to be produced'''
            if strategy=='above':
                if current_volume > mean_volume:
                    boolean_dic['{}'.format(inputSymbols[i])] = 1.0
                else:
                    boolean_dic['{}'.format(inputSymbols[i])] = -1.0

            elif strategy=='below':
                if current_volums > mean_volume:
                    boolean_dic['{}'.format(inputSymbols[i])] =	-1.0
                else:
                    boolean_dic['{}'.format(inputSymbols[i])] =	1.0

            else:
                raise TypeError('Invalid strategy type. Please enter either "above" or "below" for strategy')

            
        return boolean_dic
        

        


class nn():
    """Class for the neural network object. This class contains functions
    to create the neural network, prepare training and testing data sets,
    and run the neural network on new data. This class requires both the
    metrics and statistics classes so that it can perform calculations and
    get metrics. It also needs a username and password for a robinhood
    account.
    """

    def __init__(self, metrics, statistics, un, pw):
        self.metrics = metrics
        self.statistics = statistics
        self.un = un
        self.pw = pw
        rh.login(un, pw)

    
    def get_tt_data(self, inputSymbols, num_bands=1, interval='day', span='year', ma_short=25, ma_long=200, rsi_cutoff=30, rsi_strategy='below', days_before=10, percent_gained=4.0, percent_training=80):
        """Function that creates testing and training data sets and
        then saves them to self.x_train, self.y_train,
        self.x_test, self.y_test.

        Parameters
        inputSymbols - Give a list of stock tickers that should be used
            to create training and testing data. This should be a long
            list (at least 500 to 1000) to get a good neural network going.
        num_bands - Give an integer between 0 and 4 that will be used
            as the cutoff for bbands_bottom
        interval - Give the length of time you want the
            bollinger bands to be calculated over. Default 
            is one day
        span - Give the span of time which the averages and
            standard deviations should be calculated over.
            Default is one year. Options are week, month,
            3month, year, or 5year
        ma_short - Give an integer which represents the short
            time period moving average you want to calculate for
            ma_crossover metric in days.
        ma_long - Give an integer which represents the long
            time period moving average you want to calculate for
            ma_crossover metric in days.
        rsi_cutoff - Give a cutoff value for the relative strength
            index. This should be a real number between 0 and 100.
        rsi_strategy - Give a string that is either 'below' or 'above'.
            This tells the function to generate a buy signal when
            the stock goes either above or below the rsi_cutoff.
        days_before - Give an integer between 1 and the number of
            days that you are collecting data for each stock (controlled
            by the parameter span). This parameter tells the algorithm
            when to evaluate whether the buy signal was correctly
            or incorrectly signalled. If days_before=10,
            for example, then that means the algorithm will calculate
            the metrics and then make a prediction (once we train it)
            which will then be compared to the expected (or wanted)
            value which is just a yes or no based on whether
            the stock went up by percent_gained at any point within
            days_before of the data ending. Default for days_before
            is 10.
        percent_gained - Give a float greater than zero which represents
            the percentage gained on a stock before we give the sell
            signal. Default is 4.0
        percent_training - Give a float between 0 and 100 that represents
            the percent of the data you want to be used for training

        Returns
        Nothing. Just assigns values to self.testing and self.training
        appropriately
        """

        '''Calculate metrics for all of the inputSymbols and organize
        them in a list of zeros and ones where 0's indicate the output
        of each metric test'''
        num_metrics = 4
        tt_data = np.zeros([len(inputSymbols), num_metrics+1])

        '''loop over the stocks we're taking data from (probably
        the S&P500 or something like that)'''
        flag = False
        for i in np.arange(len(inputSymbols)):
            
            '''get stock data and calculate all the relevant metrics'''
            stock_data = rh.stocks.get_stock_historicals(inputSymbols=str(inputSymbols[i]), interval=interval, span=span)
            bbands_bottom = self.metrics.bbands_bottom(stock_data=stock_data[:-days_before], num_bands=1)
            ma_crossover = self.metrics.ma_crossover(stock_data=stock_data[:-days_before], short_period=ma_short, long_period=ma_long)
            rsi_crossover = self.metrics.rsi_crossover(stock_data=stock_data[:-days_before], rsi_cutoff=rsi_cutoff, strategy=rsi_strategy)
            volume_crossover = self.metrics.volume_crossover(stock_data=stock_data[:-days_before], time=100, strategy='above')

            '''Append to the tt_data a numpy array which looks like
            (bbands_bottom, ma_crossover, rsi_cutoff, ...) where each of these
            is just a one or zero'''
            tt_data[i,:-1] = np.array([bbands_bottom[inputSymbols[i]], ma_crossover[inputSymbols[i]], rsi_crossover[inputSymbols[i]], volume_crossover[inputSymbols[i]]])

            '''loop over days between today and the last data taken for
            calculating the metrics. Then check if the stock has risen
            percent_gained above where it was when metrics were calculated.
            Then append to tt_data as necessary.'''
            for j in np.arange(days_before):

                if float(stock_data[-1-j]['high_price']) > float(stock_data[-1-days_before]['close_price']) + float(stock_data[-1-days_before]['close_price']) * (percent_gained/100):
                    tt_data[i,-1] = 1.0
                    flag = True
                else:
                    pass

            '''if we get all the way through the loop and flag is still
            false, then that means the stock never rose percent_gained
            above where it was when metrics were calculated. So we add a
            zero to output_dict.'''
            if flag==False:
                tt_data[i,-1] = 0.0
            else:
                pass

            flag = False

            '''print progress message'''
            print('completed collecting data for {} ({} of {} stocks done)'.format(inputSymbols[i],i+1,len(inputSymbols)))

        '''Set self.x_train, self.y_train, self.x_test, and self.y_test
        to their appropriate values. Take the first percent_training percent
        of the data to use for training and the rest for testing. Also save
        stock tickers for use in other functions'''
        self.training = tt_data[:int(len(inputSymbols)*(percent_training/100)), :]
        self.testing = tt_data[int(len(inputSymbols)*(percent_training/100)):, :]
        self.test_tickers = inputSymbols[int(len(inputSymbols)*(percent_training/100)):]
        self.train_tickers = inputSymbols[:int(len(inputSymbols)*(percent_training/100))]


    def shuffle_data(self, data):
        """Function to shuffle data. The purpose of this function is to 
        shuffle training and testing data when it is fed into the network.
        The data will be shuffled along the first axis (the 0th axis in 
        python indexing).

        Parameters:
        data - Give a numpy array of any size. This is the data to be
            shuffled

        Returns
        A numpy array of the same shape as the input data but with the 
        data shuffled along the first axis.
        """

        np.random.shuffle(data)

        return data


    def build_network(self, hidden_layers=np.array([32, 16, 8]), optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError()):
        """Builds neural network with number of hidden layers equal to the
        length of hidden_layers. Uses the length of one piece of training
        data in order to determine the shape of the input layer. Because
        of this, only run this function after running get_tt_data

        Parameters
        hidden_layers - Give a 1-D numpy array whose length is the number
            of hidden layers you want and whose values are the output
            size of each layer. Default is one hidden layer with output
            size 32
        optimizer - Give a string that is the tensorflow name of the 
            optimizer you want to use. You can also pass an instance
            of tf.keras.optimizers.Optimizer
        loss - Give a string that is the tensorflow name of the loss
            function you want to use. You can also pass an instance
            of tf.keras.losses.Loss
        metrics - Give a tensorflow metrics object which is the metric
            used by the model to test its accuracy

        Returns
        Nothing. This does, however, set self.model equal to the tensorflow
        model created by this function.
        """

        '''Set up the neural network architecture.Here we use a ReLU
        activation function on the last layer.'''
        n_metrics = int(len(self.training[0,:]) - 1)
        
        self.model = tf.keras.models.Sequential()
        #Add input layer
        self.model.add(tf.keras.Input(shape=(n_metrics,)))
        #Add hidden layers with number of neurons corresponding
        #to each value in hidden_layers
        for i in np.arange(len(hidden_layers)):
            self.model.add(tf.keras.layers.Dense(hidden_layers[i]))

        #create custom activation function
        clipped_relu = lambda x: relu(x, threshold=0, max_value=1)
        
        #Add output layer that just gives one number
        self.model.add(tf.keras.layers.Dense(1))

        #Add activation layer using clipped_relu function
        self.model.add(tf.keras.layers.Activation(clipped_relu))
        
        #compile the neural network with its activation & loss functions
        self.model.compile(optimizer=optimizer, loss=loss)


    def train_network(self, batch_size=8, epochs=50):
        """Function that takes the model built by build_network
        and trains it using the training data built by get_tt_data.
        This function requires that it knows the training data as
        well as the network architecture. Therefore, it should be
        run only after both the functions get_tt_data and build_network
        have been run. This function also runs the model on the testing
        data to see how well it works on that.

        Parameters
        batch_size - Give an integer that is the size of your batches
            when training the model
        epochs - Give an integer that is the number of epochs you want
            to train your model with

        Returns
        The output of the model.fit function. This is a History
        object. The History.history attribute is a record of
        training loss values and metric values at successive
        epochs
        """

        #Shuffle training data
        shuffled_training = self.shuffle_data(data=self.training)
        x_train = shuffled_training[:,:-1]
        y_train = shuffled_training[:,-1]

        #Shuffle testing data
        shuffled_testing = self.shuffle_data(data=self.testing)
        x_test = shuffled_testing[:,:-1]
        y_test = shuffled_testing[:,-1]
        validation_data = (x_test, y_test)
        
        #Fit the model using training and testing data
        History = self.model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_data=validation_data)

        return History


    def predict(self, stock, interval='day', span='year', num_bands=1, ma_short=25, ma_long=200, rsi_cutoff=30, rsi_strategy='below', days_before=10):
        """Function that makes a buy/not buy signal prediction for
        the stock based on the model. This can only be run after
        get_tt_data, build_network, and train_network have been run. 
        The function first calculates all the metrics in the self.metrics 
        class. Then it puts the vector of 1's and 0's representing the 
        result of these metrics through the neural network.

        Parameters
        stock - Give a string that is the stock ticker which
            you want to make a prediction for. This should be
            one single ticker
        interval - Give the length of time you want the
            metrics  to be calculated over. Default
            is one day
        span - Give the span of time which the averages and
            standard deviations should be calculated over.
            Default is one year. Options are week, month,
            3month, year, or 5year
        ma_short - Give an integer which represents the short
            time period moving average you want to calculate for
            ma_crossover metric in days. This must be shorter (in days)
            than whatever span is.
        ma_long - Give an integer which represents the long
            time period moving average you want to calculate for
            ma_crossover metric in days. This must be shorter (in days)
            than whatever span is so that we have access to data
            far enough back to calculate the moving average
        rsi_cutoff - Give a cutoff value for the relative strength
            index. This should be a real number between 0 and 100.
        rsi_strategy - Give a string that is either 'below' or 'above'.
            This tells the function to generate a buy signal when
            the stock goes either above or below the rsi_cutoff.
        days_before - Give an integer that is the number of days
            that you want to allow the stock to gain money before
            testing for an accurate prediction of the algorithm

        Returns
        The output of model.predict(). This is just a float
        between 0 and 1 with 1 corresponding to a buy signal
        and 0 corresponding to a not buy signal.
        """

        stock_data = rh.stocks.get_stock_historicals(inputSymbols=stock, interval=interval, span=span)

        #Calculate metrics
        bbands_bottom = self.metrics.bbands_bottom(stock_data=stock_data[:-days_before], num_bands=1)
        ma_crossover = self.metrics.ma_crossover(stock_data=stock_data[:-days_before], short_period=ma_short, long_period=ma_long)
        rsi_crossover = self.metrics.rsi_crossover(stock_data=stock_data[:-days_before], rsi_cutoff=rsi_cutoff, strategy=rsi_strategy)
        volume_crossover = self.metrics.volume_crossover(stock_data=stock_data[:-days_before], time=100, strategy='above')

        #input_data must be passed in batches. Even though here we are
        #only using one batch, we still have to double bracket the data
        input_data = np.array([[bbands_bottom[stock[0]], ma_crossover[stock[0]], rsi_crossover[stock[0]], volume_crossover[stock[0]]]])
        prediction = self.model.predict(input_data)

        return prediction


    def test_accuracy(self):
        """Tests the accuracy of the neural network using testing
        data. Gives output in the form of percentage of stocks
        that it correctly predicted would go up in a time period
        given to the get_tt_data function.
        """

        correct=0
        incorrect=0

        k=0
        l=0
        for i in np.arange(len(self.test_tickers)):
            prediction = self.predict(stock=[str(self.test_tickers[i])])
            if prediction==0:
                binary_prediction=0
            else:
                binary_prediction=1
                k+=1

                
            if binary_prediction==1 and binary_prediction==self.testing[i,-1]:
                l+=1
            else:
                pass

                
            if binary_prediction == self.testing[i,-1]:
                correct+=1

            else:
                incorrect+=1


        self.accuracy = '{}/{} correctly predicted'.format(correct, correct+incorrect)
        print("")
        print("")
        print('{}/{} correctly predicted'.format(correct, correct+incorrect))
        print("")
        print("")
        print("Of the {} stocks in the training data, {} of our predictions were correct.".format(correct+incorrect, correct))



class transactions():
    """transactions object used in conjunction with the neural
    network (nn) object. Includes functions for buying/selling
    stocks and evaluating the neural network on stock data.
    Requires a neural network to be passed to it
    """

    def __init__(self, nn): 
        self.nn = nn

        
    def buy(self, stock, dollars):
        """Buys fractional shared of stock in the amount
        of dollars

        Parameters
        stock - Give a string that is the stock ticker
            you want to purchase
        dollars - The amount of stock you want to buy in 
            dollars. Give a float
        """
        rh.login(self.nn.un, self.nn.pw)
        stock_price = rh.get_latest_price(inputSymbols=stock)
        stock_price = float(stock_price[0])
        quantity = dollars/stock_price
        rh.orders.order_buy_fractional_by_quantity(symbol=stock, quantity=quantity)


    def sell(self, stock, percentage):
        """Function that sells the stock specified by stock.
        It sells percentage specified in the inputs of your
        total holdings in the stock
        
        Parameters
        stock - Give a string that is the stock ticker which
            you want to sell
        percentage - Give a float between 0 and 100 which indicates
            what percentage of your holdings you want to sell in
            this stock
        
        Returns
        Nothing, just sells the stock in the percentage specified
        """
        portfolio = rh.account.build_holdings()
        
        quantity = portfolio[stock]['quantity']
        quantity = (percentage/100) * quantity
        rh.orders.sell_fractional_by_quantity(symbol=stock, quantity=quantity)
        

    def sell_portfolio_at_percentage(self, percent):
        """Goes through entire portfolio and sells every stock
        which has gained percentage specified by percent since
        it was originally bought.

        Parameters
        percent - Give a float greater than 0. This specifies
            the percentage at which you want to sell all your stock

        Returns
        Nothing, just goes through the portfolio and sells if the
        stock has gained a percentage greater than percent
        """
        portfolio = rh.account.build_holdings()

        for key in portfolio.keys():
            if float(portfolio[key]['percentage']) > percent:
                self.sell(stock=key, percentage=100)
