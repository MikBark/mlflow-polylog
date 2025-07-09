"""This module provides a type-based mapping utility for Python types.

The module defines a TypeMapping class that allows mapping Python types to values,
enabling value retrieval based on the runtime type of a given key. It enforces
constraints to prevent subclass relationships among type keys, ensuring unambiguous
lookups. The module also defines a custom exception for ambiguous key matches.
"""

from collections.abc import Iterator, Mapping
from typing import Any, TypeVar

ValueType = TypeVar('ValueType')


class MultipleTypeKeysError(KeyError):
    """Raise when a key matches multiple type keys in TypeMapping.

    This exception is raised if a lookup key is an instance of multiple type keys
    in the mapping, which creates ambiguity in value retrieval.
    """

    def __init__(self, key_to_find: Any, finded_keys: list[type]) -> None:
        """Initialize MultipleTypeKeysError with the ambiguous key and matched types.

        Args:
            key_to_find : The object that was being looked up.
            finded_keys : List of type keys that matched the lookup key.
        """
        self.key_to_find = key_to_find
        self.finded_keys = finded_keys
        super().__init__(f'Key {key_to_find} is instance of many type keys {finded_keys}')

    def __repr__(self) -> str:
        """Return the string representation of the exception.

        Returns:
            A string describing the ambiguous key and the matching type keys.
        """
        return f'Key {self.key_to_find} is instance of many type keys {self.finded_keys}'


class TypeMapping(Mapping[type, ValueType]):
    """Implement a read-only mapping from types to values with isinstance-based lookup.

    TypeMapping stores pairs of Python types and associated values. When queried
    with a key, it returns the value for the first type key for which
    `isinstance(key, type_key)` is True. The mapping enforces that no two type
    keys are in a subclass relationship to prevent ambiguous lookups.

    Example usage:
        >>> tm = TypeMapping({int: "integer", str: "string"})
        >>> tm[5]
        'integer'
        >>> tm["abc"]
        'string'
    """

    def __init__(self, *initial_mappings: Mapping[type, ValueType]) -> None:
        """Initialize the TypeMapping with a dictionary of type keys and values.

        Ensures that all keys are types and that no two type keys are in a
        subclass relationship, raising an error if these constraints are violated.

        Args:
            *initial_mappings : Mapping of type keys to associated values. All keys must
                be Python types. No key may be a subclass or superclass of any other key.

        Raises:
            TypeError : If any key in init_mapping is not a Python type.
        """
        mapping = {}
        for initial_mapping in initial_mappings:
            for key, value in initial_mapping.items():
                if not isinstance(key, type):
                    raise TypeError(f'Key {key} must be a type')

                mapping[key] = value

        self._map = mapping

    def __getitem__(self, key: Any) -> ValueType:
        """Return the value for the type key matching the given key's type.

        Iterates through stored type keys and returns the value associated with
        the first type key for which `isinstance(key, type_key)` is True.

        Args:
            key : The object whose type will be checked against stored type keys.

        Returns:
            The value associated with the matching type key.

        Raises:
            KeyError : If no stored type key matches the type of the provided key.
            MultipleTypeKeysError : If the key matches multiple type keys.
        """
        keys = []
        values = []
        for type_key, value in self._map.items():
            if isinstance(key, type_key):
                keys.append(type_key)
                values.append(value)

        if len(keys) == 0:
            raise KeyError(f'No type found for instance {key} of type {type(key)}')
        if len(keys) > 1:
            raise MultipleTypeKeysError(key, keys)
        return values[0]

    def __iter__(self) -> Iterator[type]:
        """Return an iterator over the stored type keys.

        Returns:
            An iterator over the type keys in the mapping.
        """
        return iter(self._map.keys())

    def __len__(self) -> int:
        """Return the number of type-value pairs in the mapping.

        Returns:
            The number of entries in the mapping.
        """
        return len(self._map)

    def __repr__(self) -> str:
        """Return the string representation of the TypeMapping.

        Returns:
            A string representation showing the stored type-value pairs.
        """
        return f'TypeMapping({dict(self._map)})'
