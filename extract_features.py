import pandas as pd
from tsfresh import extract_features as extract_features_tsfresh
from tsfresh.utilities.dataframe_functions import roll_time_series, impute
from tsfresh.feature_selection.relevance import calculate_relevance_table

def main():
    csv_file = 'monthly_flights.csv'
    output_path = '00_raw_data/'
    window_sizes = [12]

    flight_data = pd.read_csv(output_path + csv_file, usecols=['YearMonth', 'Flights'])

    flight_features = extract_features(flight_data, output_path=output_path, csv_file=csv_file)
    return

def extract_features(flight_data, scenario_name=None, window_sizes=None, output_path=None):

    """
    Extract tsfresh features from the given time series data.

    Args:
        flight_data (pd.DataFrame): Time series data.
        scenario_name (str): Name of the scenario.
        window_sizes (list<int>): List of window sizes.
        output_path (str): Path to save the features.

    Returns:
        list<str>: List of feature file names.
        list<pd.DataFrame>: List of feature DataFrames.    
    """
    print('Extracting features...')

    # Set missing parameters to default values
    window_sizes = window_sizes or [24]
    output_path = output_path or '01_extracted_features/'
    scenario_name = scenario_name or 'scenario_name'

    # Prepare the time series
    flight_data['Symbol'] = 'TRex'
    flight_data['Time'] = range(len(flight_data))

    flight_features = list()
    for window_size in window_sizes:
        # Roll time series
        rolled_df = roll_time_series(flight_data, column_id='Symbol', column_sort='Time', max_timeshift=window_size-1)
        rolled_df = rolled_df.drop(columns=['Symbol', 'Time', 'YearMonth'])
        rolled_df['Time'] = range(len(rolled_df))

        # Extract features
        features = extract_features_tsfresh(rolled_df, 
                        column_id="id", column_sort="Time", column_value="Flights", 
                        impute_function=impute, show_warnings=False).reset_index(drop=True)
        flight_features.append(features)
    feature_file_names = list()      

    flight_features_return = flight_features.copy()
    for i in range(len(flight_features)):
        flight_features[i]['Flights'] = flight_data['Flights']
        flight_features[i]['YearMonth'] = flight_data['YearMonth']
        name = f'all_features_{scenario_name}_window-size={window_sizes[i]}.csv'
        flight_features[i].to_csv(output_path + name, index=False)
        flight_features_return[i]['Flights'] = flight_data['Flights']
        feature_file_names.append(name)
    print('Features extracted.')
    return feature_file_names, flight_features_return








if __name__ == '__main__':
    main()