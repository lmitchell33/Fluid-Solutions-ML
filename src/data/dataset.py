import torch


class DataSet(torch.utils.data.Dataset):
    '''Wrapper class used to preprocess the training data'''

    def __init__(self):
        super().__init__()