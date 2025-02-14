import numpy as np
import pandas as pd
import tensorflow as tf
import shap
from joblib import dump, load

def main():
    models_path = '02_models/'
    features_path = '01_extracted_features/'

    id = 1289

    # Import data
    features = load(models_path + f'{id}_features_original.joblib')
    model = load(models_path + f'{id}_model.joblib')

    # Summarize the background data using shap.sample or shap.kmeans
    background = shap.sample(features, 50)

    # Create the SHAP explainer with the summarized background data
    explainer = shap.KernelExplainer(model.predict, background)

    # Compute SHAP values
    shap_values = explainer.shap_values(features)

    # Plot SHAP summary plot
    shap.summary_plot(shap_values, features)

    # Calculate mean absolute SHAP values for each feature
    shap_values_abs = np.abs(shap_values)
    feature_importance = np.mean(shap_values_abs, axis=0)

    # Flatten the feature_importance array
    feature_importance = feature_importance.flatten()

    # Create a DataFrame for feature importance
    feature_importance_df = pd.DataFrame({
        'feature': features.columns,
        'importance': feature_importance
    })

    feature_importance_df['normalized_importance'] = feature_importance_df['importance'] / feature_importance_df['importance'].sum()

    # Sort and calculate cumulative importance
    feature_importance_df = feature_importance_df.sort_values(by='normalized_importance', ascending=False).reset_index(drop=True)
    feature_importance_df['cumulative_importance'] = feature_importance_df['normalized_importance'].cumsum()

    dump(feature_importance_df, features_path + f'{id}_feature_importance_df.joblib')
    dump(feature_importance, features_path + f'{id}_feature_importance.joblib')
    print('Feature importance saved successfully!')
    return


if __name__ == '__main__':
    main()