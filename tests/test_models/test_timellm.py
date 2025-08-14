import logging
import warnings

from neuralforecast.common._model_checks import check_model

# From: @marcopeix1, I think it's okay not to test TimeLLM for now.
# It needs to load an LLM, it's hard to fit in memory, it's pretty slow, so let's skip it for now.