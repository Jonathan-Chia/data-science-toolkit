import numpy as np
import pandas as pd
from sklearn.hyperparameters import GridSearchCV
from sklearn.model_selection import cross_val_score


def compare_models(_X, _y, _pipelines: dict):
    names, scores = [], []
    for name, pipeline in _pipelines.items():
        fold_scores = cross_val_score(pipeline, _X, _y, cv=5, scoring="accuracy")
        scores.append(np.mean(fold_scores))
        names.append(name)

    return pd.DataFrame({"Model": names, "Accuracy": scores})


def compare_tuned_models(_X, _y, _model_df: pd.DataFrame):
    names, scores, grid_searches = [], [], []
    for index, row in _model_df.iterrows():
        gs = GridSearchCV(
            estimator=row["pipeline"],
            param_grid=row["param_grid"],
            scoring="accuracy",
            cv=5,
        )
        gs.fit(_X, _y)
        scores.append(gs.best_score_)
        names.append(row["name"])
        grid_searches.append(gs)

    return pd.DataFrame({"Model": names, "Accuracy": scores, "GS": grid_searches})


# example usage:

# nb_model = GaussianNB()
# rf_model = RandomForestClassifier(random_state=0)
# nn_model = MLPClassifier(random_state=0)
# svc_model = SVC(class_weight='balanced', kernel='linear', C=2)
# svc_rbf_model = SVC(class_weight='balanced', kernel='rbf', C=2)

# nb_pipeline = Pipeline(steps=[('preprocessor', preprocessor_standard),
#                               ('model', nb_model)])

# rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor_minmax),
#                               ('model', rf_model)])

# nn_pipeline = Pipeline(steps=[('preprocessor', preprocessor_minmax),
#                               ('model', nn_model)])

# svc_pipeline = Pipeline(steps=[('preprocessor', preprocessor_minmax),
#                               ('model', svc_model)])

# svc_rbf_pipeline = Pipeline(steps=[('preprocessor', preprocessor_minmax),
#                               ('model', svc_rbf_model)])

# pipeline_dict = {'naive bayes': nb_pipeline,
#                  'random forest': rf_pipeline,
#                  'neural network': nn_pipeline,
#                  'svc (linear)': svc_pipeline,
#                  'svc (rbf)': svc_rbf_pipeline}

# results_df = compare_models(X, y, pipeline_dict)

# name = ['random forest', 'neural network', 'svc (linear)', 'svc (rbf)']
# pipeline = [rf_pipeline, nn_pipeline, svc_pipeline, svc_rbf_pipeline]
# param_grid = [param_grid_random_forest, param_grid_nn, param_grid_svc, param_grid_svc_rbf]

# model_df = pd.DataFrame({'name': name,
#                          'pipeline': pipeline,
#                          'param_grid': param_grid})

# model_df

# tuned_results_df = compare_tuned_models(X, y, model_df)

# feature_importances = tuned_results_df.loc[0].GS.best_estimator_.named_steps['model'].feature_importances_
# importance_df = pd.DataFrame({'Feature': tuned_results_df.loc[0].GS.best_estimator_.named_steps['preprocessor'].get_feature_names_out(), 'Importance': feature_importances})

# Sort the DataFrame by importance values (optional)
# importance_df = importance_df.sort_values(by='Importance', ascending=False)

# # Plot feature importances
# plt.figure(figsize=(8, 6))
# plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
# plt.xlabel('Feature Importance')
# plt.ylabel('Feature')
# plt.title('Random Forest Feature Importances')
# plt.gca().invert_yaxis()  # Invert the y-axis for better visualization (optional)
# plt.show()
