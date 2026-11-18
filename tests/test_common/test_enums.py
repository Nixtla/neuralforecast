from neuralforecast.common.enums import (
    ADDITIVE_EXPLAINERS,
    ALL_EXPLAINERS,
    ExplainerEnum,
)


def test_explainer_collections_use_exact_membership():
    assert ALL_EXPLAINERS == (
        ExplainerEnum.IntegratedGradients,
        ExplainerEnum.ShapleyValueSampling,
        ExplainerEnum.InputXGradient,
    )
    assert ADDITIVE_EXPLAINERS == frozenset(
        {
            ExplainerEnum.IntegratedGradients,
            ExplainerEnum.ShapleyValueSampling,
        }
    )
    assert "IntegratedGradients" in ALL_EXPLAINERS
    assert "ShapleyValueSampling" in ALL_EXPLAINERS
    assert "InputXGradient" in ALL_EXPLAINERS
    assert "IntegratedGradients" in ADDITIVE_EXPLAINERS
    assert "ShapleyValueSampling" in ADDITIVE_EXPLAINERS
    assert "InputXGradient" not in ADDITIVE_EXPLAINERS
    assert "Gradient" not in ALL_EXPLAINERS
    assert "Sampling" not in ADDITIVE_EXPLAINERS
    assert "Input" not in ALL_EXPLAINERS


def test_explainer_enum_contains_only_explainer_values():
    assert list(ExplainerEnum) == list(ALL_EXPLAINERS)
