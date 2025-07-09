import math

import pytest

from mlflow_polymodel.type_mapping import MultipleTypeKeysError, TypeMapping


@pytest.fixture
def basic_type_mapping_dict():
    return {int: 'integer', str: 'string', list: 'list'}


@pytest.fixture
def basic_type_mapping(basic_type_mapping_dict):
    return TypeMapping(basic_type_mapping_dict)


@pytest.fixture
def empty_type_mapping_dict():
    return {}


@pytest.fixture
def empty_type_mapping(empty_type_mapping_dict):
    return TypeMapping(empty_type_mapping_dict)


@pytest.fixture
def single_type_mapping():
    return TypeMapping({int: 'integer'})


@pytest.mark.parametrize(
    'mapping_dict,expected_len',
    [
        ({int: 'integer', str: 'string', list: 'list'}, 3),
        ({}, 0),
        ({str: 'string'}, 1),
    ],
)
def test_init_with_different_mappings(mapping_dict, expected_len):
    tm = TypeMapping(mapping_dict)
    assert len(tm) == expected_len


@pytest.mark.parametrize(
    'invalid_key',
    [
        'not_a_type',
        42,
        None,
        'string',
        math.pi,
    ],
)
def test_init_with_invalid_keys_raises_type_error(invalid_key):
    mapping = {invalid_key: 'value'}
    with pytest.raises(TypeError):
        TypeMapping(mapping)


@pytest.mark.parametrize(
    'test_value,expected_value',
    [
        (42, 'integer'),
        ('hello', 'string'),
        ([1, 2, 3], 'list'),
        (0, 'integer'),
        (-5, 'integer'),
        ('', 'string'),
        ([], 'list'),
        (1, 'integer'),
        (-1, 'integer'),
        (999999, 'integer'),
    ],
)
def test_getitem_with_matching_types(basic_type_mapping, test_value, expected_value):
    result = basic_type_mapping[test_value]
    assert result == expected_value


@pytest.mark.parametrize(
    'test_value',
    [
        math.pi,
        (1, 2, 3),
        {'key': 'value'},
        {1, 2, 3},
    ],
)
def test_getitem_with_unmatched_types_raises_key_error(basic_type_mapping, test_value):
    with pytest.raises(KeyError):
        basic_type_mapping[test_value]


@pytest.mark.parametrize(
    'test_value',
    [
        math.pi,
        (1, 2, 3),
    ],
)
def test_getitem_specific_errors(basic_type_mapping, test_value):
    with pytest.raises(KeyError):
        basic_type_mapping[test_value]


def test_getitem_with_empty_mapping_raises_key_error(empty_type_mapping):
    with pytest.raises(KeyError):
        empty_type_mapping[42]


def test_getitem_with_multiple_matching_types_raises_multiple_type_keys_error():
    mapping = {object: 'object', int: 'integer'}
    tm = TypeMapping(mapping)
    with pytest.raises(MultipleTypeKeysError) as exc_info:
        tm[42]

    assert exc_info.value.key_to_find == 42
    assert set(exc_info.value.finded_keys) == {object, int}


def test_getitem_with_subclass_inheritance_raises_multiple_type_keys_error():
    mapping = {Exception: 'exception', ValueError: 'value_error'}
    tm = TypeMapping(mapping)
    test_value = ValueError('test')
    with pytest.raises(MultipleTypeKeysError) as exc_info:
        tm[test_value]

    assert exc_info.value.key_to_find == test_value
    assert set(exc_info.value.finded_keys) == {Exception, ValueError}


def test_multiple_type_keys_error_attributes():
    key = 42
    keys = [object, int]
    error = MultipleTypeKeysError(key, keys)

    assert error.key_to_find == key
    assert error.finded_keys == keys


@pytest.mark.parametrize(
    'fixture_name,expected_behavior',
    [
        ('basic_type_mapping', 'contains_int'),
        ('basic_type_mapping', 'all_keys_match'),
        ('empty_type_mapping', 'empty_list'),
        ('single_type_mapping', 'single_int_key'),
    ],
)
def test_iter_behavior(fixture_name, expected_behavior, request):
    type_mapping = request.getfixturevalue(fixture_name)
    keys = list(iter(type_mapping))

    if expected_behavior == 'contains_int':
        assert int in keys
    elif expected_behavior == 'all_keys_match':
        assert set(keys) == {int, str, list}
    elif expected_behavior == 'empty_list':
        assert len(keys) == 0
    elif expected_behavior == 'single_int_key':
        assert keys == [int]


@pytest.mark.parametrize(
    'fixture_name,expected_len',
    [
        ('basic_type_mapping', 3),
        ('empty_type_mapping', 0),
        ('single_type_mapping', 1),
    ],
)
def test_len_returns_correct_size(fixture_name, expected_len, request):
    type_mapping = request.getfixturevalue(fixture_name)
    assert len(type_mapping) == expected_len


@pytest.mark.parametrize(
    'fixture_name,expected_behavior',
    [
        ('basic_type_mapping', 'starts_with_TypeMapping'),
        ('basic_type_mapping', 'contains_int'),
        ('empty_type_mapping', 'equals_empty'),
        ('single_type_mapping', 'equals_single'),
    ],
)
def test_repr_behavior(fixture_name, expected_behavior, request):
    type_mapping = request.getfixturevalue(fixture_name)
    result = repr(type_mapping)

    if expected_behavior == 'starts_with_TypeMapping':
        assert result.startswith('TypeMapping(')
    elif expected_behavior == 'contains_int':
        assert 'int' in result
    elif expected_behavior == 'equals_empty':
        assert result == 'TypeMapping({})'
    elif expected_behavior == 'equals_single':
        assert result == "TypeMapping({<class 'int'>: 'integer'})"

def test_mapping_setter_with_valid_types():
    tm = TypeMapping({})
    assert len(tm) == 2


@pytest.mark.parametrize(
    'type_key,value',
    [
        (int, 'integer'),
        (str, 'string'),
        (float, 'float'),
        (bool, 'boolean'),
    ],
)
def test_init_with_various_valid_types(type_key, value):
    mapping = {type_key: value}
    tm = TypeMapping(mapping)
    assert len(tm) == 1


@pytest.mark.parametrize(
    'empty_value',
    [
        '',
        [],
        0,
    ],
)
def test_getitem_with_empty_values_returns_correct_type(empty_value):
    mapping = {str: 'string', list: 'list', int: 'integer'}
    tm = TypeMapping(mapping)
    result = tm[empty_value]
    assert result == mapping[type(empty_value)]
