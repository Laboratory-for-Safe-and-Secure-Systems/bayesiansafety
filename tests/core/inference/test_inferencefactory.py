from unittest.mock import patch
import pytest

from bayesiansafety.core.inference import IInference
from bayesiansafety.core.inference import InferenceFactory, Backend
# fixtures provided via conftest


#################### Tests for Interface  ###########################

@patch.multiple(IInference, __abstractmethods__=set())
def test_validate_interface_instantiation_raises_exception():
    instance = IInference()

    with pytest.raises(NotImplementedError) as e:
        assert instance.query(variables=None, evidence=None)

    with pytest.raises(NotImplementedError) as e:
        assert instance.interventional_query(
            variables=None, do=None, evidence=None)

    with pytest.raises(NotImplementedError) as e:
        assert instance.counterfactual_query(
            target=None, do=None, observed=None)


@patch('bayesiansafety.core.inference.InferenceFactory.get_configured_backend', autospec=True)
@pytest.mark.parametrize("test_backend", [Backend.PGMPY, Backend.PYAGRUM])
def test_validate_engine_inst_is_valid_subclass_of_iinference_success(fn_default_backend, fixture_bn_confounder_param, test_backend):
    model_name = 'test'
    model, _ = fixture_bn_confounder_param(model_name)

    fn_default_backend.return_value = test_backend

    factory = InferenceFactory(model=model)

    fn_default_backend.assert_called_once()

    assert issubclass(type(factory.get_engine()), IInference)


################ Tests for the factory itself  ######################

@patch('bayesiansafety.core.inference.InferenceFactory._InferenceFactory__create_pgmpy_inference', autospec=True)
@patch('bayesiansafety.core.inference.InferenceFactory._InferenceFactory__create_pyagrum_inference', autospec=True)
@patch('bayesiansafety.core.inference.InferenceFactory.get_configured_backend', autospec=True)
@pytest.mark.parametrize("test_backend", [Backend.PYAGRUM, Backend.PGMPY])
def test_factory_instantiate_with_valid_backend_success(fn_default_backend, ctor_pyagrum, ctor_pgmpy, fixture_bn_confounder_param, test_backend):
    model_name = 'test'
    model, _ = fixture_bn_confounder_param(model_name)

    fn_default_backend.return_value = test_backend

    factory = InferenceFactory(model=model)

    fn_default_backend.assert_called_once()

    if test_backend is Backend.PGMPY:
        ctor_pgmpy.assert_called_once() #.assert_called_with(model=model)

    elif test_backend is Backend.PYAGRUM:
        ctor_pyagrum.assert_called_once() #.assert_called_with(model=model)



@patch('bayesiansafety.core.inference.InferenceFactory.get_configured_backend', autospec=True)
@pytest.mark.parametrize("invalid_backend", ["INVALID", "pgmpy", None, -1])
def test_factory_instantiate_with_invalid_backend_raises_exception(fn_default_backend, fixture_bn_confounder_param, invalid_backend):
    expected_exc_substring = "Invalid backend selected"
    model_name = 'test'
    model, _ = fixture_bn_confounder_param(model_name)

    fn_default_backend.return_value = invalid_backend

    with pytest.raises(Exception) as e:
        assert InferenceFactory(model=model)

    assert expected_exc_substring in str(e.value)


@patch('bayesiansafety.core.inference.InferenceFactory._InferenceFactory__create_pgmpy_inference', autospec=True)
@patch('bayesiansafety.core.inference.InferenceFactory._InferenceFactory__create_pyagrum_inference', autospec=True)
@patch('bayesiansafety.core.inference.InferenceFactory.get_configured_backend', autospec=True)
@pytest.mark.parametrize("default_backend, requested_backend", [(Backend.PGMPY, Backend.PGMPY), (Backend.PYAGRUM, Backend.PYAGRUM), (Backend.PGMPY, Backend.PYAGRUM), (Backend.PYAGRUM, Backend.PGMPY)])
def test_factory_get_engine_success(fn_default_backend, ctor_pyagrum, ctor_pgmpy, fixture_bn_confounder_param, default_backend, requested_backend):
    model_name = 'test'
    model, _ = fixture_bn_confounder_param(model_name)

    fn_default_backend.return_value = default_backend

    # here ctor is called -> reset mock
    factory = InferenceFactory(model=model)

    if default_backend != requested_backend:
        ctor_pgmpy.reset_mock()
        ctor_pyagrum.reset_mock()

    factory.get_engine(backend=requested_backend)

    if requested_backend is Backend.PGMPY:
        ctor_pgmpy.assert_called_once()

    elif requested_backend is Backend.PYAGRUM:
        ctor_pyagrum.assert_called_once()
