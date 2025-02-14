# average_impyct vs time_frame_features_size vs num_roll_features

import plotly.graph_objs as go
import numpy as np
from joblib import load

def main():
    print('Plotting average impact...')

    features_path = r'01_extracted_features/'
    models_path = r'02_models/'
    plot_path = r'08_plots_eval/'

    id = 4605 # ID of the model whose feature importance is to be plotted
    importance_path = f'{id}_feature_importance_df.joblib' # This has to be calculated using SHAP
    features_original = load(f'{models_path}{4248}_features_original.joblib') # Can be any model's features that include all the features rolled


    feature_importance = load(features_path + importance_path)
    # Filter features to only include rolled features. Take only the first of these rolled features.
    cols = list(features_original.columns) + [col + '_0' for col in features_original.columns]
    feature_importance_filtered = feature_importance[feature_importance['feature'].isin(cols)]

    # Max of range should be the maximum number of features that can be rolled
    num_roll_features = range(155)

    # Max of range should be the maximum possible time frame for feature rolling.
    # The max time frame has to be decided based on the number of maximum data points
    time_frame_features_size = range(25)
    
    # Calculate average impact for each combination of num_roll_features and time_frame_features_size
    avg_impact = np.zeros((len(num_roll_features), len(time_frame_features_size)))
    n = 0
    for i in range(len(num_roll_features)):
        for j in range(len(time_frame_features_size)):
            num = 0
            for k in range(num_roll_features[i]):
                impact_this = feature_importance_filtered.iloc[k]['normalized_importance']
                avg_impact[i, j] += impact_this * time_frame_features_size[j]
                num += time_frame_features_size[j]
                n += 1
            for k in range(len(feature_importance_filtered)-num_roll_features[i]):
                impact_this = feature_importance_filtered.iloc[k+num_roll_features[i]]['normalized_importance']
                avg_impact[i, j] += impact_this
                num += 1
                n += 1
            avg_impact[i, j] /= num

    fig = go.Figure(data=go.Heatmap(
        z=avg_impact,
        x=list(time_frame_features_size),
        y=list(num_roll_features),
        colorscale='Viridis'))
    
    fig.update_layout(
        title='Average impact vs. time_frame_features_size vs. num_roll_features',
        xaxis_title='time_frame_features_size',
        yaxis_title='num_roll_features')
    
    min_impact = np.min(avg_impact)
    min_indices = np.unravel_index(np.argmax(avg_impact), avg_impact.shape)
    min_time_frame_size = time_frame_features_size[min_indices[1]]
    min_num_roll_features = num_roll_features[min_indices[0]]
    if False:
        fig.add_trace(go.Scatter(
            x=[min_time_frame_size],
            y=[min_num_roll_features],
            mode='markers',
            marker=dict(color='red', size=10),
            name=f'Optimal Point: {min_impact:.4f}'
        ))

    fig.write_image(plot_path + f'{id}_average_impact_plot.png', format='png', width=1920, height=1080)
    fig.write_image(plot_path + f'{id}_average_impact_plot.svg', format='svg', scale=3)
    

    fig.show()
    fig.write_html(plot_path + f'{id}_average_impact_plot.html')

    print('Plot saved.')
    return








if __name__ == '__main__':
    main()