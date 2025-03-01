from torch.nn import Module, LSTM, Linear

class PredictorLSTM(Module):
    """
    A class to represent the LSTM used to predict blood volume status.
    
    This class implements a Long Short-Term Memory (LSTM) network with configurable 
    input size, hidden layer size, number of layers, and sequence length. It is designed 
    to process patient vitals over time and predict the current blood volume status.

    Methods:
        forward(x): Performs the forward pass through the LSTM model and returns the output.
    """
    def __init__(self, input_size, hidden_size, num_layers, sequence_len, num_classes=3):
        '''
        Args:
            input_size {int} -- The number of features/parameters per time step (this is the number of values inputted into the model). (Heart Rate, CVP, etc...)
            hidden_size {int} -- The number of neurons in each layer. (Capacity of the model to learn patterns)
            num_layers {int} -- Number of stacked layers. Higher number of layers increases ability to learn complex patterns.
            sequence_len {int} -- The number of time steps in the sequence. (number of steps to look back in time at)
        
        Kwargs:
            num_classes {int} -- Number of output classes. Default is 3: low, normal, high.
        '''
        super(PredictorLSTM, self).__init__()
        
        # initalize the variables to be used throughout the rest of the class
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_len = sequence_len

        # initialize the LSTM model with given parameters.
        # batch_first=True means the input tensor has the shape (batch_size, seq_len, input_size),
        self.lstm = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # initialize the fully connected layer to output the final prediction.
        self.fc = Linear(hidden_size, num_classes)

    def forward(self, x):
        '''This function is the forward pass of the model and should return the output of the model.
        TODO: The contents of this function is dependent on the number of layers we want and the amount of data we have to train the model.
        '''
        pass


if __name__ == "__main__":
    # if calling the file directly, print the details of the model.
    LSTM = PredictorLSTM(7, 128, 3)
    print(LSTM)