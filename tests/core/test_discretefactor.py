import pytest
import numpy as np
np.random.seed(42)

from bayesiansafety.core import DiscreteFactor
# fixtures provided via conftest


@pytest.mark.parametrize("name  ,  scope            , cardinalities, values         , state_names",
                         [("", ["A"], [2], np.array([0.8, 0.2]), {}),
                          ("Test", ["A", "B", "C"], [2, 4, 7], np.ones(2*4*7), {}),
                          ("Test", ["A", "B"], [2, 3], np.zeros(2*3), {"A": ["s_a1", "s_a2"], "B":[0, 1, 2]})])
def test_instantiate_success(name, scope, cardinalities, values, state_names):
    factor = DiscreteFactor(
        name=name, scope=scope, cardinalities=cardinalities, values=values, state_names=state_names)

    assert factor.name == name
    assert factor.scope == scope
    assert factor.cardinalities == cardinalities
    assert np.allclose(factor.values, values.reshape(cardinalities), atol=0.0)
    assert factor.state_names == state_names


def test_copy_succes():
    scope = ["A", "B", "C"]
    cards = [2, 4, 7]
    state_names = dict.fromkeys(scope, [])
    for var, card in zip(scope, cards):
        for cnt in range(card):
            state_names[var].append(f"{var}_{cnt}")

    factor = DiscreteFactor(name="Test", scope=scope, cardinalities=cards,
                            values=np.ones(2*4*7), state_names=state_names)

    copied_factor = factor.copy()

    assert factor.name == copied_factor.name
    assert factor.scope == copied_factor.scope
    assert factor.cardinalities == copied_factor.cardinalities
    assert np.allclose(factor.values, copied_factor.values, atol=0.0)
    assert factor.state_names == state_names


def test_get_probabilities_returns_correct():
    scope = ["A", "B", "C"]
    cards = [2, 4, 7]
    state_names = dict.fromkeys(scope, [])
    for var, card in zip(scope, cards):
        for cnt in range(card):
            state_names[var].append(f"{var}_{cnt}")

    values = np.random.rand(2*4*7)
    factor = DiscreteFactor(name="Test", scope=scope,
                            cardinalities=cards, values=values, state_names=state_names)

    assert np.allclose(factor.get_probabilities().reshape(cards), values.reshape(cards), atol=1e-32)


def test_get_value_returns_correct():
    scope = ["A", "B", "C"]
    cards = [2, 2, 2]
    state_names = {"A": ["s_a1", "s_a2"], "B": [
        "s_b1", "s_b2"], "C": ["s_c1", "s_c2"]}
    values = [0, 1, 2, 3, 4, 5, 6, 7]
    combinations = [({"A": "s_a1", "B": "s_b1", "C": "s_c1"}, 0),
                    ({"A": "s_a1", "B": "s_b1", "C": "s_c2"}, 1),
                    ({"A": "s_a1", "B": "s_b2", "C": "s_c1"}, 2),
                    ({"A": "s_a1", "B": "s_b2", "C": "s_c2"}, 3),
                    ({"A": "s_a2", "B": "s_b1", "C": "s_c1"}, 4),
                    ({"A": "s_a2", "B": "s_b1", "C": "s_c2"}, 5),
                    ({"A": "s_a2", "B": "s_b2", "C": "s_c1"}, 6),
                    ({"A": "s_a2", "B": "s_b2", "C": "s_c2"}, 7)]

    factor = DiscreteFactor(name="Test", scope=scope,
                            cardinalities=cards, values=values, state_names=state_names)

    for state_combo, expected_val in combinations:
        assert np.isclose(factor.get_value(state_combo), expected_val, atol=0.0)


@pytest.mark.parametrize("bad_combination", [{"A": "s_a1", "B": "s_b1"},  {"A": "s_a1", "B": "s_b2", "C": "s_c0", "D": "s_d1"}])
def test_get_value_wrong_number_of_query_states_raises_exception(bad_combination):
    expected_exc_substring = "do not match with the factors named vars"
    scope = ["A", "B", "C"]
    cards = [2, 2, 2]
    state_names = {"A": ["s_a1", "s_a2"], "B": [
        "s_b1", "s_b2"], "C": ["s_c1", "s_c2"]}
    values = [0, 1, 2, 3, 4, 5, 6, 7]

    factor = DiscreteFactor(name="Test", scope=scope,
                            cardinalities=cards, values=values, state_names=state_names)

    with pytest.raises(ValueError) as e:
        assert factor.get_value(bad_combination)

    assert expected_exc_substring in str(e.value)


@pytest.mark.parametrize("bad_combination", [{"A": "s_a0", "B": "s_b1", "C": "s_c2"}, {"A": "s_a1", "B": "s_b1", "C": 0}])
def test_get_value_invalid_state_name_raises_exception(bad_combination):
    expected_exc_substring = "are not part of the listed state names"

    scope = ["A", "B", "C"]
    cards = [2, 2, 2]
    state_names = {"A": ["s_a1", "s_a2"], "B": [
        "s_b1", "s_b2"], "C": ["s_c1", "s_c2"]}
    values = [0, 1, 2, 3, 4, 5, 6, 7]

    factor = DiscreteFactor(name="Test", scope=scope,
                            cardinalities=cards, values=values, state_names=state_names)

    with pytest.raises(ValueError) as e:
        assert factor.get_value(bad_combination)

    assert expected_exc_substring in str(e.value)
