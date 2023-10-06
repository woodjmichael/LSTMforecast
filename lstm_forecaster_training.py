# import os, sys

from datetime import datetime

import numpy as np
import pandas as pd
import emd

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from pickle import dump, load

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from PROPHET_DB import mysql

import argparse
import configparser
import sys
from datetime import datetime, timedelta
from pathlib import Path

# pd.options.plotting.backend = "plotly"
# pd.set_option('precision', 2)


# pd.options.plotting.backend = "plotly"
# pd.set_option('precision', 2)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred)))

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="EMS predictive optimizer")
    parser.add_argument("config", help="configuration file")
    return parser.parse_args(argv)


def read_config(config_file):
    config = configparser.ConfigParser()
    config.sections()
    config_file_fullname = Path(config_file)
    if config_file_fullname.exists():
        config.read(config_file)
    else:
        # LOGGER.error(f"Not possible to access scheduler configuration file")
        sys.exit(1)
    return config

def create_query_test(talktoSQL, table_mysql, column_mysql, time_column, days):
    query = "SELECT {0}, {1} FROM {2}.{3} " \
            "WHERE({0} <= NOW() AND {0} >= DATE_SUB(NOW(), INTERVAL '{4}' DAY))".format(time_column, column_mysql, talktoSQL._database, table_mysql, days)
    return query

class RunTheJoules:
    def __init__(self,
                 site,
                 filename,
                 models_dir=r'./models/',
                 data_points_per_day=96,
                 persist_days=1,
                 data_cols=[1],
                 resample='15min',
                 rename=True,
                 remove_days=None):
        self.site = site
        self.data_points_per_day = data_points_per_day
        self.persist_days = persist_days
        self.persist_lag = data_points_per_day * persist_days
        self.models_dir = models_dir
        self.filename = filename
        self.data_cols = data_cols
        self.resample = resample
        self.remove_days = remove_days
        self.df = self.get_dat(rename)

    def emd_sift(self, df):
        imf = emd.sift.sift(df['Load (kW)'].values)

        for i in range(imf.shape[1]):
            df['IMF%s' % (i + 1)] = imf[:, i]

        return df

    def get_dat(self,rename):
        df = pd.read_csv(self.filename,
                         comment='#',
                         parse_dates=True,
                         index_col=0,
                         usecols=[0] + self.data_cols)

        if rename:
            df.columns = ['Load (kW)']

        # df = df.tz_localize('Etc/GMT+8',ambiguous='infer') # or 'US/Eastern' but no 'US/Pacific'
        df = df.resample('15min').mean()
        # df = df.tz_convert(None)
        df = df.fillna(method='ffill').fillna(method='bfill')

        # df.loc['2020-12-8'] = df['2020-12-1'].values
        # df.loc['2020-11-26'] = df['2020-11-19'].values
        # # df.loc['2021-10-11':'2021-10-15'] = df['2021-10-17':'2021-10-'].values
        # df.loc['2022-5-24'] = df['2022-5-17'].values

        if self.remove_days == 'weekends':
            df = df[df.index.weekday < 5]
        elif self.remove_days == 'weekdays':
            df = df[df.index.weekday >= 5]

        #df = self.emd_sift(df)

        # df['Day'] = df.index.dayofyear
        # df['Hour'] = df.index.hour
        # df['Weekday'] = df.index.dayofweek

        df['Persist'] = df['Load (kW)'].shift(self.persist_lag)

        df = df.fillna(method='bfill')

        df = df[:'2022-08']

        return df

    def get_dat_test(self, filename, features):
        df = pd.read_csv(filename,
                         comment='#',
                         parse_dates=True,
                         index_col=0,
                         usecols=[0] + self.data_cols, )

        df.columns = ['Load (kW)']

        # df = df.tz_localize('Etc/GMT+8',ambiguous='infer') # or 'US/Eastern' but no 'US/Pacific'
        df = df.resample('15min').mean()
        # df = df.tz_convert(None)
        df = df.fillna(method='ffill').fillna(method='bfill')

        # df.loc['2020-12-8'] = df['2020-12-1'].values
        # df.loc['2020-11-26'] = df['2020-11-19'].values
        # #df.loc['2021-10-11':'2021-10-15'] = df['2021-10-17':'2021-10-'].values
        # df.loc['2022-5-24'] = df['2022-5-17'].values

        if self.remove_days == 'weekends':
            df = df[df.index.weekday < 5]
        elif self.remove_days == 'weekdays':
            df = df[df.index.weekday >= 5]

        df = self.emd_sift(df)

        # df['Day'] = df.index.dayofyear
        # df['Hour'] = df.index.hour
        # df['Weekday'] = df.index.dayofweek

        df['Persist'] = df['Load (kW)'].shift(self.persist_lag)

        df = df.fillna(method='bfill')

        # df = df[:'2022-08']

        return df[features]

    """
    Generator function for creating random batches of training-data.
    """

    def train_batch_generator(self, batch_size, L_sequence_in, L_sequence_out, n_x_signals, n_y_signals, n, x, y):

        # Infinite loop.
        while True:
            # Allocate a new array for the batch of input-signals.
            x_shape = (batch_size, L_sequence_in, n_x_signals)
            x_batch = np.zeros(shape=x_shape, dtype=np.float16)

            # Allocate a new array for the batch of output-signals.
            y_shape = (batch_size, L_sequence_out, n_y_signals)
            y_batch = np.zeros(shape=y_shape, dtype=np.float16)

            # Fill the batch with random sequences of data.
            for i in range(batch_size):
                # Get a random start-index.
                # This points somewhere into the training-data.
                idx = np.random.randint(n - L_sequence_in - L_sequence_out)

                # Copy the sequences of data starting at this index.
                x_batch[i] = x[idx:idx + L_sequence_in]
                y_batch[i] = y[idx:idx + L_sequence_out]

            yield (x_batch, y_batch)

    def organize_dat_v5(self, df, n_in, n_out, train_split, batch_size):

        # scalers
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()

        # shift for forecast
        # shift_steps = 1 * 24 * 4    # Number of time steps
        df_targets = df['Load (kW)'].shift(-1 * n_in)

        # scale and adjust the length

        x_data = x_scaler.fit_transform(df.values)[0:-1 * n_in]
        y_data = y_scaler.fit_transform(df_targets.values[:-1 * n_in, np.newaxis])
        # y_data = np.expand_dims(y_data,axis=1)

        dump(x_scaler, open(self.models_dir + "x_scaler.pkl", 'wb'))
        dump(y_scaler, open(self.models_dir + "y_scaler.pkl", 'wb'))

        num_data = len(x_data)
        num_train = int(train_split * num_data)
        num_test = num_data - num_train

        x_train = x_data[0:num_train]
        x_test = x_data[num_train:]
        len(x_train) + len(x_test)

        y_train = y_data[0:num_train]
        y_test = y_data[num_train:]
        len(y_train) + len(y_test)

        num_x_signals = x_data.shape[1]
        num_y_signals = y_data.shape[1]

        generator = self.train_batch_generator(batch_size,
                                               n_in,
                                               n_in,  # note! this is not a typo
                                               num_x_signals,
                                               num_y_signals,
                                               num_train,
                                               x_train,
                                               y_train)

        # x_batch, y_batch = next(generator)

        test_data = (np.expand_dims(x_test, axis=0), np.expand_dims(y_test, axis=0))

        return (num_x_signals, num_y_signals, generator, test_data, y_scaler)

    def organize_dat_test(self, df):

        x_scaler = load(open(self.models_dir + "x_scaler.pkl", 'rb'))

        x_test = x_scaler.transform(df.values)

        return np.expand_dims(x_test, axis=0)

    def train_lstm_v4(self, n_features_x: int, n_in: int, n_out: int, path_checkpoint: str,
                      generator, validation_data, units: list, epochs: int,
                      layers: int = 1, patience: int = 5, verbose: int = 1, dropout: list = None,
                      afuncs={'lstm': 'relu', 'dense': 'sigmoid'},
                      loss='mse', ):

        model = Sequential()
        # model.add( LSTM(    units,
        #                                     return_sequences=True,
        #                                     input_shape=(None, num_x_signals,)))
        model.add(LSTM(units[0],
                       return_sequences=True,
                       input_shape=(None, n_features_x,),
                       activation=afuncs['lstm']))
        if dropout is not None:
            model.add(Dropout(dropout[0]))
        if (layers == 2) and (len(units) > 1):
            # model.add( LSTM(    units,
            #                                     return_sequences=True) ) 
            model.add(LSTM(units[1],
                           return_sequences=True,
                           activation=afuncs['lstm']))
            if dropout is not None:
                model.add(Dropout(dropout[1]))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss=loss, optimizer='adam')
        model.summary()

        callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                              monitor='val_loss',
                                              verbose=verbose,
                                              # save_weights_only=True,
                                              save_best_only=True)

        callback_early_stopping = EarlyStopping(monitor='val_loss',
                                                patience=patience,
                                                verbose=verbose,
                                                restore_best_weights=True)

        callback_tensorboard = TensorBoard(log_dir='./logs/',
                                           histogram_freq=0,
                                           write_graph=False)

        # callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
        #                                                                         factor=0.1,
        #                                                                         min_lr=1e-4,
        #                                                                         patience=0,
        #                                                                         verbose=verbose)

        callbacks = [callback_early_stopping,
                     callback_checkpoint,
                     callback_tensorboard, ]
        # callback_reduce_lr]

        hx = model.fit(x=generator,
                       epochs=epochs,
                       steps_per_epoch=100,
                       validation_data=validation_data,
                       callbacks=callbacks,
                       verbose=verbose)

        return model, hx

    def plot_training_history(self, hx):
        hx_loss = hx.history['loss']
        hx_val_loss = hx.history['val_loss']
        t = np.arange(len(hx_loss))

        plt.figure(figsize=(10, 6))
        plt.plot(t, hx_loss, t, hx_val_loss)
        plt.legend(['Train Set Loss', 'Test Set Loss'])
        plt.ylabel('MSE (scaled data) [kW/kW]')
        plt.xlabel('Training Epoch')
        plt.title('Training History')
        plt.show()

    def plot_predictions_week(self, y_true, y_pred, week=0, ppd=96, persist_days=7):

        ppw = ppd * 7  # points per week, points per day
        ppp = persist_days * ppd
        t = np.arange(ppw)

        y_true_wk = y_true[(week + 1) * ppw:(week + 2) * ppw]
        y_np1w_wk = y_true[(week + 1) * ppw - ppp:(week + 2) * ppw - ppp]
        y_pred_wk = y_pred[(week + 1) * ppw:(week + 2) * ppw]

        rmse_np1w = np.sqrt(np.mean(np.square(y_np1w_wk - y_true_wk)))
        rmse_pred = np.sqrt(np.mean(np.square(y_pred_wk - y_true_wk)))
        skill = (rmse_np1w - rmse_pred) / rmse_np1w

        plt.figure(figsize=(20, 6))
        plt.plot(t, y_true_wk,
                 t, y_np1w_wk,
                 t, y_pred_wk)

        plt.legend(['true',
                    f'np 1 wk (rmse {rmse_np1w:.0f})',
                    f'predict (rmse {rmse_pred:.0f})'])

        plt.title(f'test set week {week + 1} of {int(len(y_true) / ppw)} '
                  f'(skill {skill:.2})')
        plt.show()

    def plot_one_prediction(self, model, x_test, y_test, n_in, n_out, begin=0):

        y_pred = model.predict(x_test[:, begin:begin + n_in, :])

        t_in = np.arange(n_in)
        t_out = np.arange(n_in, n_in + n_out)

        for z in range(x_test.shape[2]):
            plt.plot(t_in, x_test[0, begin:begin + n_in, z], label=f'x test f{z}')
        plt.plot(t_out, y_test[0, begin:begin + n_out, 0], label='y test')
        plt.plot(t_out, y_pred[0, :n_out, 0], '--', label='y pred')
        plt.legend()
        plt.show()

    def run_them_fast(self, features, units, dropout, n_in, n_out, epochs=100, patience=10, verbose=0,
                      output=False, plots=False, train_split=0.9, batch_size=256):
        layers = len(units)

        df = self.df[features]

        # header             
        units_str = ''
        for u in units:
            units_str += (str(u) + ' ')

        print(f'\n\n////////// units={units_str} layers={layers} //////////\n')

        # meta
        y, m, d = datetime.now().year - 2000, datetime.now().month, datetime.now().day
        path_checkpoint = self.models_dir + f'lstm.keras'

        (n_features_x, n_features_y,
         batchgen, dat_valid,
         scaler) = self.organize_dat_v5(df, n_in, n_out, train_split, batch_size)

        (x_test, y_test) = dat_valid

        # np
        #y_test_naive_mse = naive_forecast_mse(y_test[0, :, 0], horizon=self.persist_lag)

        # model                                                
        self.model, hx = self.train_lstm_v4(n_features_x, n_in, n_out,
                                            path_checkpoint, batchgen,
                                            dat_valid, units, epochs,
                                            layers, patience,
                                            verbose, dropout)

        # evaluate
        y_test_predict = self.model.predict(x_test)

        y_test_pred_kw = scaler.inverse_transform(y_test_predict[:, :, 0]).flatten()
        y_test_kw = scaler.inverse_transform(y_test[:, :, 0]).flatten()

        test_rmse_np = rmse(y_test_kw[self.persist_lag:],
                            y_test_kw[:-(self.persist_lag)])

        results = {}
        results['test_rmse_pred'] = rmse(y_test_kw, y_test_pred_kw)
        results['test_skill'] = 1 - results['test_rmse_pred'] / test_rmse_np
        results['test_std_diff_pred'] = np.diff(y_test_pred_kw).std()
        results['epochs'] = len(hx.history['loss']) - patience

        if output:
            print('test set')
            print(f'rmse np     {test_rmse_np:.2f}')
            print(f'rmse pred {rmse(y_test_kw, y_test_pred_kw):.2f}')
            print(f'skill         {1 - rmse(y_test_kw, y_test_pred_kw) / test_rmse_np:.3f}')

        # plot
        if plots:
            self.plot_training_history(hx)
            self.plot_predictions_week(y_test_kw, y_test_pred_kw, week=0)
            self.plot_one_prediction(self.model, x_test, y_test, n_in, n_out)

        return results, hx.history

    def banana_clipper(self, t_now, plot=False):
        df = self.get_dat_test(
                                #filename=r'/home/mjw/OneDrive/Data/Load/Vehicle/ACN/test_JPL_v2.csv',
                                filename=r'D:\GitHub\SANDBOX3\data\test_JPL_v2.csv',
                               features=['Load (kW)', 'Persist'] + [f'IMF{x}' for x in range(3, 7)]
        )

        x = df[:t_now - pd.Timedelta('15min')][-96:].copy()
        y = df[['Load (kW)']][t_now:][:96].copy()

        # print('\n\n\nx',x, len(x),'\n\n\n')
        # print('\n\n\ny',y, len(y),'\n\n\n')

        x_test = self.organize_dat_test(x)

        y_scaler = load(open(self.models_dir + "y_scaler.pkl", 'rb'))

        model = tf.keras.models.load_model(self.models_dir + "lstm.keras")

        y_pred = model.predict(x_test[:, -96:, :])

        y_pred_kw = y_scaler.inverse_transform(y_pred[:, :, 0]).flatten()

        if plot:
            plt.figure(figsize=(10, 5))
            plt.plot(df['Load (kW)'][:t_now - pd.Timedelta('15min')][-96:], label='x load')
            plt.plot(df['Persist'][:t_now - pd.Timedelta('15min')][-96:], label='x persist')
            plt.plot(y, label='y true')
            plt.plot(y.index, y_pred_kw.flatten(), label='y pred')
            plt.legend()
            plt.show()


def main(argv=None):
    # args = parse_arguments(argv)
    #
    # config = read_config(args.config)
    # data_opt = {
    #     'n_back': config.getint("data_opt", "n_back"),  # 4*24*7
    #     'n_timesteps': config.getint("data_opt", "n_timesteps"),  # 4*4
    #     'lag': config.getint("data_opt", "lag"),
    #     'tr_per': config.getfloat("data_opt", "tr_per"),
    #     'out_col': config.get("data_opt", "out_col").split(','),
    #     'features': config.get("data_opt", "features").split(','),
    #     'freq': config.getint("data_opt", "freq"),
    #     'tr_days_step': config.getint("data_opt", "tr_days_step"),
    # }
    # if data_opt['features'] == ['']:
    #     data_opt['columns'] = data_opt['out_col']
    # else:
    #     data_opt['columns'] = data_opt['features'] + data_opt['out_col']
    # data_opt['n_features'] = len(data_opt['columns'])
    #
    # model_opt = {'Dense_input_dim': config.getint("model_opt", "Dense_input_dim"),
    #              'LSTM_num_hidden_units': list(map(int, config.get("model_opt", "LSTM_num_hidden_units").split(','))),
    #              'LSTM_layers': config.getint("model_opt", "LSTM_layers"),
    #              'metrics': config.get("model_opt", "metrics"), 'optimizer': config.get("model_opt", "optimizer"),
    #              'patience': config.getint("model_opt", "patience"), 'epochs': config.getint("model_opt", "epochs"),
    #              'validation_split': config.getfloat("model_opt", "validation_split"),
    #              'model_path': config.get("model_opt", "model_path"),
    #              'Dropout_rate': config.getfloat("model_opt", "Dropout_rate"),
    #              'input_dim': (data_opt['n_back'], data_opt['n_features']), 'dense_out': data_opt['n_timesteps']
    #              }
    #
    # talktoSQL = mysql.MySQLConnector(database=config["mysql"]["database"],
    #                                  host=config["mysql"]["host"],
    #                                  port=config["mysql"]["port"],
    #                                  user=config["mysql"]["user"],
    #                                  password=config["mysql"]["password"])
    #
    # days_back = 10
    # # ev_query = util.create_query(talktoSQL, config["sql_table"]["ev_power_table"], config["sql_table"]["time_column"])
    # ev_query_test = create_query_test(talktoSQL, config["sql_table"]["ev_power_table"],
    #                                        config["sql_table"]["ev_power_field"],
    #                                        config["sql_table"]["time_column"], days_back)
    # df = talktoSQL.read_query(ev_query_test, {config["sql_table"]["time_column"]})

    jpl = RunTheJoules('acn-jpl',
                       #models_dir='./models/acn-jpl/',
                       models_dir=r'D:/GitHub/SANDBOX3/SANDBOX3/models/acn-jpl/',
                       # filename='C:/Users/Admin/OneDrive - Politecnico di Milano/Data/Load/Vehicle/ACN/train_JPL_v2.csv'
                       #filename='/home/mjw/OneDrive/Data/Load/Vehicle/ACN/train_JPL_v2.csv',
                       filename=r'D:\GitHub\SANDBOX3\data\train_JPL_v2_weekdays.csv',
                       #remove_days='weekends',
                       rename=False,
                       data_cols=list(range(1,13))
                       )

    # lstm.plot_weekly_overlaid(jpl.df,days_per_week=5)

    results, history = jpl.run_them_fast(features=['Load (kW)', 'Persist'],#+ [f'IMF{x}' for x in range(3, 7)],
                                         units=[12,12],#[48, 512],
                                         dropout=2 * [0.2],
                                         n_in=96,
                                         n_out=96,
                                         epochs=10,#100,
                                         patience=15,
                                         plots=True,
                                         output=True,
                                         verbose=1)

    #jpl.banana_clipper(pd.Timestamp('2023-10-2 0:15'), plot=True)

    # rx = {}
    # hx = {}
    # for units1 in [24,48,96,128,256,512]:
    #     for units2 in [24,48,96,128,256,512]:
    #         for dropout in [0., 0.2]:
    #             for n_in in [96,2*96,3*96]:
    #                 model_name = f'lstm u1{units1}u2{units2}d{dropout}n{n_in}'                
    #                 results, history = jpl.run_them_fast(features=['Load (kW)','Persist'] + [f'IMF{x}' for x in range(3,7)],
    #                                                     units=[units1,units2],dropout=2*[dropout],n_in=n_in,n_out=96,
    #                                                     epochs=100,patience=15,
    #                                                     plots=False,output=True,verbose=0)
    # rx = pd.DataFrame(rx).transpose()
    # rx.to_csv(jpl.model_dir+'results.csv')
    # rx

if __name__ == '__main__':
    main()
