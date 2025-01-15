import torch


class DataSet(torch.utils.data.Dataset):
    '''Wrapper class used to preprocess the training data'''
    # TODO: Update this class based on the data we want to use

    def __init__(self):
        super(DataSet, self).__init__()


    def __getitem__(self, index):
        return super().__getitem__(index)