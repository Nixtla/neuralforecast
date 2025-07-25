# Univariate models:
  - Explain future_exog, hist_exog_cols
  - Pass whatever data they want
  - Modify 
    - core: we define the explain method (look into predict), this returns the dict.
    - base_model: create the wrapper of the model, explanation function, naming the features.
  - # Create pytest for models: NHITS, LSTM with recurrent=True, 
# NHITS and NBEATS may need a special treatment for the forward method.
Delete from common.base_model:
explain_prediction
_create_shap_wrapper
_aggregate_shap_by_feature
_parse_shap_input
_get_feature_names
_extract_target_features
Exposed to user: core nf.explain()