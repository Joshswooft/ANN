import numpy as np


class NeuralNetwork:
        
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learn_rate = 0.1, layers = 2):
        self.learn_rate = learn_rate
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.iterations = 1000
        self.layers = layers
        self.initWeights()
        self.norm_params_set  = False

    '''
    Train fn()
    fwd_pass
    backpass
    update_weights
    '''
    #sigmoid function = 1/1+e^-x
    #gives us probability as an output
    def sigmoid(self, x, deriv=False):
        if(deriv == True):
            return x * (1-x)    #return slope
        return 1/(1 + np.exp(-x))

    def set_weights(self, who, wih):
        self.W_hidden_output = who
        self.W_input_hidden = wih
    
    def initWeights(self):
        '''
        create random initial weights in range -1 to 1
        Normal distribution -1/sqr(n) to 1/sqr(n) where n = # of links into a node
        '''
        #lambda creates annoymnous fn, with input x
        sqrt = lambda x: x**(1/2.0)
        
        if self.layers == 1:
            self.W_input_hidden = np.random.normal(0.0, sqrt(1/self.input_nodes), (self.output_nodes, self.input_nodes))
            print ('wih shape: ', self.W_input_hidden.shape)
        
        if self.layers > 1:
            #create weight matrix from input node to hidden. Shape = (num hidden, num of features)
            self.W_input_hidden = np.random.normal(0.0, sqrt(1/self.input_nodes), (self.hidden_nodes, self.input_nodes))
        
            #weight matrix from hidden -> output. Shape = (num outputs, num hidden)
            self.W_hidden_output = np.random.normal(0.0, sqrt(1/self.hidden_nodes), (self.output_nodes, self.hidden_nodes))
    
    #can also use this function to test how accurate neural network is by passing in test data
    def fwd_pass(self, trainInputs = []):
        
        if len(trainInputs) == 0:
            trainInputs = self.inputs
        
        if self.layers == 1:
            self.X_output = self.sigmoid( np.dot( trainInputs, self.W_input_hidden ), False)
        
        if self.layers == 2:      
            #multiply the input layer by the weight matrix and then use the activation fn (e.g. sigmoid)
            self.X_hidden = self.sigmoid( np.dot( self.W_input_hidden, trainInputs ), False )
            #Xhidden shape = (num hidden, num of inputs) e.g. (200, 1443)
            self.X_output = self.sigmoid( np.dot( self.W_hidden_output  , self.X_hidden ), False )
            #TODO: change this code so it can be adapted for more 'layers' of hidden nodes e.g. I -> h1 -> h2 -> O

        #X_output is array of size 1 x num input nodes        
        #returns an array of size 1 x num input nodes
        y = self.denormalize(self.X_output)
        return y
    
    def denormalize(self, z):
        '''
        z = normalized var
        return the original X
        '''
        X = ( (self.maxY - self.minY) * (z - self.a) / (self.b - self.a) ) + self.minY
        return X

    
    def back_pass(self):
        
        output_errs = self.outputs - self.X_output

        #note: 1 layer network is basically X -> Y  
        if self.layers == 1:
            hidden_errs = output_errs * self.sigmoid(self.X_output, True)
            self.W_input_hidden = self.W_input_hidden + self.learn_rate * ( np.dot( self.inputs.T, hidden_errs )  )
            
        if self.layers > 1:
            sig_errs = output_errs *self.sigmoid( self.X_output , True)
            hidden_errs = np.dot( self.W_hidden_output.T, output_errs )
            #update weights
            self.W_hidden_output = self.W_hidden_output + self.learn_rate * ( np.dot( sig_errs, self.X_hidden.T ) )
            self.W_input_hidden = self.W_input_hidden + self.learn_rate * (np.dot(  (hidden_errs * self.sigmoid(self.X_hidden, True) ), self.inputs.T ) )

    def train(self, inputs, outputs):
        self.inputs = np.array(inputs, ndmin=2, dtype=np.float32).T
        self.outputs = np.array(outputs, ndmin=2, dtype=np.float32).T
        self.fwd_pass()
        
        #get error and then update weights
        try:
            if self.norm_params_set == False:
                raise ValueError('normalization parameters not set')                        
            self.back_pass()
        except ValueError as e:
            print ("ERROR: ", e)
            raise
            
        
    def copy(self):
        '''return a new neural network with the exact same variables'''
        obj = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes, self.learn_rate, self.layers)
        obj.set_weights(self.W_hidden_output, self.W_input_hidden)
        return obj

    def set_norm_params(self, maxY, minY, a, b):
        '''
        sets the normalization parameters so we know how to normalize/denormalize
        helper function for calculating the error
        '''
        self.maxY = maxY
        self.minY = minY
        self.a = a
        self.b = b
        self.norm_params_set = True
        
    def get_Y(self):
        return self.denormalize(self.X_output)
    
