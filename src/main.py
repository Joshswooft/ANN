import data_preprocessing
import analysis
import neural_network
import evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


def apply_training(nn):
    epochs = 50
    prev_err = 9999   #some really large number
    model_err = 999
    errs = []
    total_epochs = 0
    cycles = []

    #returns an array of the normalized features
    scaler = data_preprocessing.get_scaler()
    norm_x = scaler.fit_transform(x)
    #array of normalized predictand
    norm_y = scaler.fit_transform(y)
    max_y = scaler.data_max_[-1]
    min_y = scaler.data_min_[-1]
    nn.set_norm_params(max_y, min_y, 0.1, 0.9 )
    err_threshold = 0.03

    while abs(prev_err + err_threshold) > abs(model_err) and total_epochs < 5000:
        if prev_err == 100:
            prev_err = 10
        else:
            prev_err = model_err
        
        rmse = []           
        for iter in range(0, epochs):
            yp = []
            for t in range (0, len(x)):
                nn.train(norm_x[t], norm_y[t])
                yp.append(nn.get_Y().item() )
                # yp.append(np.asscalar(nn.get_Y() ) )
            #calculate err
            r = mean_squared_error(y, yp) ** 0.5
            rmse.append(r)
        plt.plot(rmse)   
        #how to see what each colour means??
        total_epochs += epochs
        cycles.append(total_epochs)    
        
        #discard the model afterwards
        test_model = nn.copy()
        #different kf for each round to measure stability
        kf = evaluation.cross_val(x, y)
        model_err = evaluation.cross_validation(kf, test_model, x, y)
        print ('if prev err < err:', prev_err, model_err)
        errs.append(abs(model_err))


if __name__ == "__main__":
    df = data_preprocessing.import_data()
    print(df.head())

    df = data_preprocessing.clean_errorenous_data(df)
    # analysis.plot_every_col(df)

    df = data_preprocessing.remove_outliers(df)

    # analysis.plot_every_col(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(numeric_cols)
    df = data_preprocessing.normalize_df(df[numeric_cols])
    features = ['T', 'W', 'SR', 'DSP', 'DRH']
    inputs_nn = df[features]    #returns data frame with the cols we will use as features
    output = df[['PanE']]

    x, x_test, y, y_test = train_test_split(inputs_nn, output, test_size=0.1)     #90% TRAIN

    input_nodes = len(x)
    output_nodes = 1#len(y) #was 1
    hidden_nodes = 5  #just a rand number
    print ('o_shape: ', output.shape)
    layers = 2
    learn_rate = 0.1
    nn = neural_network.NeuralNetwork(inputs_nn.shape[1], hidden_nodes, output_nodes, learn_rate, layers)
    apply_training(nn)