from sklearn.model_selection import KFold
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def cross_val(X, y, splits= 10):
    kf = KFold(n_splits=splits, shuffle=True) # Define the split - into 3 folds - more folds, longer it will take
    return kf.split(X,y) #split this into train, test sets


def cross_validation(kf, model, x, y): 
    '''
    splits the data into k folds
    trains the data on k-1 folds and tests on the kth model
    does this k times
    sums the error for each fold and returns the avg
    '''
    model_err = []
    bias = 100
    #go through each fold in kf
    for k, (train, test) in enumerate(kf):
        y_pred = []
        train_data = np.array(x)[train]
        y_train = np.array(y)[train]
        test_data = np.array(x)[test]
        y_test = np.array(y)[test]
    
        #normalization should occur in here so we don't information leak
        #any fitting should be done in the folds - https://gerardnico.com/data_mining/cross_validation#cross-validation_error_estimate
        
        
        maxY = max(y_test)
        minY = min(y_test)
        a = 0.1
        b = 0.9
        #norm
        #z = [a + ( (x - minY)*(b-a) / (maxY - minY) ) for x in y]
        
        #have to scale it to range 0.1 - 0.9 so we avoid multiplying weights by 0
        scaler = MinMaxScaler(feature_range=(0.1,0.9))

        norm_train_data = scaler.fit_transform(train_data) #arr containing normalized features
        norm_y_train = scaler.fit_transform(y_train)
        norm_test_data = scaler.fit_transform(test_data)
    
        
        #now train using this data - take avg of errors
        try:
            model.set_norm_params(maxY, minY, a, b)
            for t in range(0, len(train)):
                model.train(norm_train_data[t], norm_y_train[t])
    
            for t in range(0, len(test)):
                yp = model.fwd_pass( np.array(norm_test_data[t], ndmin=2).T) 
                yp = yp.item()
                # yp = np.asscalar(yp)
                y_pred.append(yp)
                           
            #mean squared error for each fold
            mse_fold = mean_squared_error(y_test, y_pred)
            rmse = mse_fold**0.5
            model_err.append(rmse) 
            #take avg error of each fold 
        except ValueError as e:
            print(e)
        
    print ('fold errors: ', model_err)
    try:   
        bias = sum(model_err) / len(model_err)  #bias = avg error
        print ('bias: ', bias)
    except ZeroDivisionError as e:
        print (e)

    #variance = std dev of all errors
# =============================================================================
#     variance = sum([(error - mean)**2 for error in errors])/(k)
#     standardDeviation = variance**.5
#     confidenceInterval = (mean - 1.96*standardDeviation, mean + 1.96*standardDeviation)
# =============================================================================
    return bias
