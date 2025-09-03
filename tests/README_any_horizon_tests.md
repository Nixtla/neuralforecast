# Any Horizon Predictions Test Suite

This test suite validates the any horizon predictions functionality added in the `feat/any_horizon_predictions_test` branch. The feature allows neuralforecast models to make predictions for horizons longer than the trained horizon.

## Overview

The any horizon predictions feature enables models to:
- Make predictions beyond their trained horizon length
- Maintain prediction consistency across different horizon lengths
- Handle exogenous variables correctly during extended predictions
- Work with both recurrent and non-recurrent models

## Test Structure

### 1. Dummy Models (`test_dummy_models.py`)

**Purpose**: Test the core any horizon functionality with simple, predictable models.

**Models**:
- `DummyUnivariate`: Simple naive prediction model
- `DummyRNN`: Seasonal naive prediction model with RNN structure
- `DummyMultivariate`: Multivariate seasonal naive prediction model

**Key Tests**:
- Basic any horizon predictions (h > trained_h)
- Consistency across different horizon lengths
- Integration with NeuralForecast
- Error handling for invalid horizons
- Memory efficiency for long horizons

### 2. Any Horizon Predictions (`test_any_horizon_predictions.py`)

**Purpose**: Test any horizon functionality with existing neuralforecast models.

**Models Tested**:
- RNN, LSTM, GRU (recurrent models)
- TCN, DeepAR (recurrent models)
- MLP (non-recurrent model)

**Key Tests**:
- Basic any horizon predictions
- Consistency across different horizon lengths
- Integration with NeuralForecast
- Different loss functions (MAE, MSE, MQLoss)
- Edge cases (very short/long horizons)
- Memory efficiency

### 3. NeuralForecast Integration (`test_neuralforecast_any_horizon.py`)

**Purpose**: Test any horizon functionality through the NeuralForecast interface.

**Key Tests**:
- Multiple models with different horizons
- Consistency across different horizon lengths
- Different loss functions and quantile outputs
- Multiple time series
- Step size variations
- Error handling

### 4. Exogenous Variables (`test_any_horizon_exogenous.py`)

**Purpose**: Test any horizon functionality with various types of exogenous variables.

**Exogenous Types Tested**:
- Future exogenous variables (`futr_exog_list`)
- Historical exogenous variables (`hist_exog_list`)
- Static exogenous variables (`stat_exog_list`)
- Mixed exogenous variables

**Key Tests**:
- Any horizon predictions with future exogenous
- Any horizon predictions with historical exogenous
- Any horizon predictions with static exogenous
- Mixed exogenous scenarios
- Consistency across horizon lengths
- Memory efficiency

## Test Scenarios

### Basic Functionality
- Train model with horizon h
- Predict with horizon h (baseline)
- Predict with horizon 2h, 3h, etc.
- Predict with horizon not multiple of h (e.g., h=5, predict h=17)

### Consistency Tests
- First h predictions should be identical across different horizon lengths
- Predictions should be deterministic (same input → same output)
- Model state should reset properly after predictions

### Edge Cases
- Very short input sizes
- Very long horizons (10x+ trained horizon)
- Horizon equal to input size
- Zero or negative horizons (error handling)

### Exogenous Variables
- Future exogenous available for extended horizon
- Historical exogenous handling
- Static exogenous integration
- Mixed exogenous scenarios

### Memory and Performance
- No memory leaks during long horizon predictions
- Efficient handling of very long horizons
- Proper cleanup after predictions

## Running the Tests

### Prerequisites
- Python 3.8+
- PyTorch
- pandas, numpy, pytest
- neuralforecast (with any horizon feature)

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Files
```bash
# Test dummy models
pytest tests/test_models/test_dummy_models.py -v

# Test any horizon predictions
pytest tests/test_any_horizon_predictions.py -v

# Test NeuralForecast integration
pytest tests/test_neuralforecast_any_horizon.py -v

# Test exogenous variables
pytest tests/test_any_horizon_exogenous.py -v
```

### Run Specific Test Classes
```bash
# Test dummy models
pytest tests/test_models/test_dummy_models.py::TestDummyModels -v

# Test any horizon predictions
pytest tests/test_any_horizon_predictions.py::TestAnyHorizonPredictions -v
```

### Run Specific Test Methods
```bash
# Test basic functionality
pytest tests/test_models/test_dummy_models.py::TestDummyModels::test_dummy_univariate_basic -v

# Test consistency
pytest tests/test_any_horizon_predictions.py::TestAnyHorizonPredictions::test_any_horizon_consistency -v
```

## Test Data

The test suite uses synthetic data with the following characteristics:

### Simple Data
- Single time series with trend + noise
- 200 daily observations
- Deterministic (seeded random)

### Multivariate Data
- Two time series with different patterns
- Trend + noise vs. sinusoidal + noise
- Same time range and frequency

### Exogenous Data
- Future exogenous: trend, seasonal, event indicators
- Historical exogenous: lagged values, trend
- Static exogenous: categorical, regional, size features

## Expected Behavior

### Recurrent Models
- Should extend predictions beyond trained horizon
- Should maintain prediction quality
- Should handle exogenous variables correctly
- Should reset state after predictions

### Non-Recurrent Models
- Should use recursive prediction for extended horizons
- Should maintain prediction consistency
- Should handle exogenous variables correctly

### General Requirements
- No NaN predictions
- Consistent predictions across horizon lengths
- Proper error handling for invalid inputs
- Memory efficient for long horizons

## Debugging

### Common Issues
1. **Model not extending horizon**: Check if `RECURRENT=True` and `h_train` is set correctly
2. **Exogenous variable errors**: Verify exogenous data availability for extended horizon
3. **Memory issues**: Check for proper state reset in recurrent models
4. **Inconsistent predictions**: Verify prediction logic in model's forward method

### Debug Mode
Run tests with increased verbosity and output:
```bash
pytest tests/ -v -s --tb=long
```

### Specific Model Debugging
```bash
# Test specific model with debug output
pytest tests/test_any_horizon_predictions.py::TestAnyHorizonPredictions::test_rnn_any_horizon_basic -v -s
```

## Contributing

When adding new tests:

1. **Follow naming conventions**: `test_<feature>_<scenario>`
2. **Use descriptive test names**: Clear about what is being tested
3. **Include edge cases**: Test boundary conditions and error scenarios
4. **Add documentation**: Clear docstrings explaining test purpose
5. **Maintain consistency**: Follow existing test patterns and structure

## Future Enhancements

Potential areas for additional testing:

1. **Performance benchmarks**: Measure prediction time vs. horizon length
2. **Accuracy degradation**: Test prediction quality for very long horizons
3. **Batch processing**: Test with multiple series and batch predictions
4. **Distributed training**: Test with multi-GPU scenarios
5. **Model serialization**: Test save/load with extended horizon predictions
