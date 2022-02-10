import pytest
from bayesiansafety.faulttree import Cutset

#fixtures provided via conftest

def test_get_minimal_cuts_wrong_algorithm_raises_exception(fixture_ft_and_only_model):
    bad_algorithm = "bad"
    expected_exc_substring = "Unsupported algorithm"
    model, _ = fixture_ft_and_only_model

    with pytest.raises(Exception) as e:
        assert Cutset(model).get_minimal_cuts(algorithm=bad_algorithm)

    assert expected_exc_substring in str(e.value)


@pytest.mark.parametrize("algorithm", ["mocus","fatram"])
def test_cutsets_and_only_model(algorithm, fixture_ft_and_only_model):
    model, correct_cutsets = fixture_ft_and_only_model
    candidate_cutsets = Cutset(model).get_minimal_cuts(algorithm=algorithm)

    corr = set([frozenset(elem) for elem in correct_cutsets])
    mins = set([frozenset(elem) for elem in candidate_cutsets])
    assert len(corr^mins)  == 0


@pytest.mark.parametrize("algorithm", ["mocus","fatram"])
def test_cutsets_or_only_model(algorithm, fixture_ft_or_only_model):
    model, correct_cutsets = fixture_ft_or_only_model
    candidate_cutsets = Cutset(model).get_minimal_cuts(algorithm=algorithm)

    corr = set([frozenset(elem) for elem in correct_cutsets])
    mins = set([frozenset(elem) for elem in candidate_cutsets])
    assert len(corr^mins)  == 0


@pytest.mark.parametrize("algorithm", ["mocus","fatram"])
def test_cutsets_fatram_paper_model(algorithm, fixture_ft_fatram_paper_model):
    model, correct_cutsets = fixture_ft_fatram_paper_model
    candidate_cutsets = Cutset(model).get_minimal_cuts(algorithm=algorithm)

    corr = set([frozenset(elem) for elem in correct_cutsets])
    mins = set([frozenset(elem) for elem in candidate_cutsets])
    assert len(corr^mins)  == 0


@pytest.mark.parametrize("algorithm", ["mocus","fatram"])
def test_cutsets_mocus_book_model(algorithm, fixture_ft_mocus_book_model):
    model, correct_cutsets = fixture_ft_mocus_book_model
    candidate_cutsets = Cutset(model).get_minimal_cuts(algorithm=algorithm)

    corr = set([frozenset(elem) for elem in correct_cutsets])
    mins = set([frozenset(elem) for elem in candidate_cutsets])
    assert len(corr^mins)  == 0


@pytest.mark.parametrize("algorithm", ["mocus","fatram"])
def test_cutsets_elevator_model(algorithm, fixture_ft_elevator_model):
    model, correct_cutsets = fixture_ft_elevator_model
    candidate_cutsets = Cutset(model).get_minimal_cuts(algorithm=algorithm)

    corr = set([frozenset(elem) for elem in correct_cutsets])
    mins = set([frozenset(elem) for elem in candidate_cutsets])
    assert len(corr^mins)  == 0
