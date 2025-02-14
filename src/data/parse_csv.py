import os
import pandas as pd

'''
Update these funcitons to better parse data from the excel sheets.
Need to get more ids for the vitals we want from the d_items file.
Then we can use that to get a list of chartevents. Then we can use 
that to find the fluid inputs and potentially the fluid outputs. 
'''


def get_items(features, input_file=None, output_file=None):
    '''Parses a csv file and finds the correct data for a Mimic dataset'''

    file_path = input_file or os.path.expanduser('~/Fluid-Solutions-ML/data/raw/d_items.csv')
    output_path = output_file or os.path.expanduser('~/Fluid-Solutions-ML/data/processed/item_names.csv')
    
    df = pd.read_csv(file_path) 
    mask = df.apply(lambda row: any(feature in str(value).lower() for value in row for feature in features), axis=1)
    filtered_df = df[mask].drop_duplicates(subset=['itemid'])
    filtered_df.to_csv(output_path, sep=',', index=False, header=['itemid', 'label', 'linksto'], columns=['itemid', 'label', 'linksto'], lineterminator='\n')

    return filtered_df['itemid']


def get_chart_events(items, input_file=None, output_file=None):
    file_path = input_file or os.path.expanduser('~/Fluid-Solutions-ML/data/raw/chartevents.csv')
    output_path = output_file or os.path.expanduser('~/Fluid-Solutions-ML/data/processed/events.csv')

    df = pd.read_csv(file_path)
    events = df.loc(df['itemid'].isin(items))


def get_fluid_inputs():
    '''im not entirely sure if this is true, but should I basically just go through each of the chartevents
    then find a fluid input that matches the chartevent. 
    '''
    pass


if __name__ == "__main__":
    features = ['central venous pressure', 'mean arterial pressure', 'spo2', 'ppv', 'blood pressure', 'heart rate', 'lactate']
    items = get_items(features)

    get_chart_events(list(items))

    # get_items(['fluid'], output_file=f"{os.path.expanduser('~/Fluid-Solutions-ML/data/processed/fluid_events.csv')}")
    # get_chart_events(items)