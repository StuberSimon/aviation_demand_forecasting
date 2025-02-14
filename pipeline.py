from long_term_model_script import long_term_forecast
from Train_model_script_v1 import train_models
from forecast_plot import plot_forecast
from joblib import load
from hmape_input_size_plot import plot_hmape_input_size
from sbs_model_train_script import train_models_sbs
from hmape_covid_impact_plot import plot_covid_impact
from extract_features import extract_features
import pandas as pd

# If set to True, the pipeline will train new models.
# If set to False, the pipeline will use the model IDs provided in the model_ids list.
train_new_models = True

# If set to True, the pipeline will create new forecasts.
# If set to False, the pipeline will use the model IDs provided in the forecasted_ids list.
do_new_forecast = True 
do_covid_impact = True
change_input = True
model_ids = [1] # If train_new_models is set to False, the pipeline will use the models with IDs provided in this list.
forecasted_ids = [] # If do_new_forecast is set to False, the pipeline will use the models with IDs provided in this list.
summary_path = 'Summary_new.xlsx'

# CSV file containing the time series data for the monthly number of flights.
# Has to contain cols 'YearMonth' and 'Flights'. The index will be ignored.
input_file = 'monthly_flights.csv'  

# Use this to specify the scenario name for the feature files.
# If not specified, the default value is 'scenario_name'.
# If you want to train a model with default parameters and different input data, change this name.
# Otherwise, the pipeline will not train new models.
scenario_name = 'pacific_islands_1' 

# Please close the Excel file (summary) before running the pipeline.
def pipeline():    
    # You can manually import new flight data or features here.
    # If you want the pipeline to read the data from the input file, leave these as None.
    flight_data = None
    features_list = None
    features_names_list = None

    if change_input:
        if flight_data is None and input_file is not None:
            flight_data = pd.read_csv('00_raw_data/' + input_file, usecols=['YearMonth', 'Flights'])
        if features_list is None or features_names_list is None:
            # Specify tsfresh window sizes while calling the function if needed. Specifying more than one window size will create multiple feature DataFrames.
            features_names_list, features_list = extract_features(flight_data, scenario_name=scenario_name)

    if train_new_models:
        # Specify model parameter while calling the function if needed
        new_ids = train_models(summary_path=summary_path, features_list=features_list, feature_file_names=features_names_list)
    elif len(model_ids) > 0:
        new_ids = model_ids
    else:
        print('No models to train. Exiting pipeline.')
        return
        
    if do_covid_impact:
        # Specify model parameter while calling the function if needed
        sbs_ids = train_models_sbs(model_ids=new_ids, summary_path=summary_path)
        covid_impact_plot_file = plot_covid_impact(sbs_ids, summary_path=summary_path)

    if do_new_forecast:
        long_term_ids = long_term_forecast(new_ids, summary_path=summary_path)
    elif len(forecasted_ids) > 0:
        long_term_ids = forecasted_ids
    else:
        print('No forecast to plot. Exiting pipeline.')
        return
    
    forecast_file = plot_forecast(long_term_ids, summary_path=summary_path)

    hmape_input_size_plot_file = plot_hmape_input_size(long_term_ids, summary_path=summary_path)






    print('Pipeline completed. Forecast file saved to: ')
    print(forecast_file)
    print('Covid impact plot file saved to: ')
    print(covid_impact_plot_file)
    print('HMAPE input size plot file saved to: ')
    print(hmape_input_size_plot_file)    
    return

if __name__ == '__main__':
    pipeline()