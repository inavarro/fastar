#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Registry for Initial Mass Functions (IMFs).
Provides centralized access to all available IMF parametrizations.
"""

from fastar.imf.named_imf.single_power_law import single_power_law
from fastar.imf.named_imf.broken_power_law import broken_power_law
from fastar.imf.named_imf.kroupa import kroupa
from fastar.imf.named_imf.chabrier import chabrier
from fastar.imf.named_imf.flexi import flexi


class IMFRegistry:
    """
    Registry for Initial Mass Functions (IMFs).

    This class provides a centralized way to access pre-defined IMF
    parametrizations by name, and allows registration of custom IMFs.

    Supports multiple access patterns:
    - Method: imf_registry.load_by_name('kroupa')
    - Dict-like: imf_registry['kroupa']
    - Attribute: imf_registry.kroupa

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from fastar.imf import imf_registry
    >>>
    >>> # Method access
    >>> imf_func = imf_registry.load_by_name('kroupa')
    >>>
    >>> # Dict-like access
    >>> imf_func = imf_registry['kroupa']
    >>>
    >>> # Attribute access
    >>> imf_func = imf_registry.kroupa
    """

    def __init__(self):
        """
        Initialize the IMF registry with pre-defined IMFs.
        """
        self._registry = {
            'single_power_law': single_power_law,
            'broken_power_law': broken_power_law,
            'kroupa': kroupa,
            'chabrier': chabrier,
            'flexi': flexi,
        }

    def load_by_name(self, name):
        """
        Load an IMF function by name.

        Parameters
        ----------
        name : str
            Name of the IMF to load. Available options are:
            - 'single_power_law'
            - 'broken_power_law'
            - 'kroupa'
            - 'chabrier'
            - 'flexi'

        Returns
        -------
        callable
            The IMF function with signature `func(mass, params)`.

        Raises
        ------
        ValueError
            If the IMF name is not found in the registry.

        Examples
        --------
        >>> from fastar.imf import imf_registry
        >>> kroupa_imf = imf_registry.load_by_name('kroupa')
        >>> mass = jnp.array([0.5, 1.0, 2.0])
        >>> imf_vals = kroupa_imf(mass, {'m_min': 0.1, 'm_max': 100.0})
        """
        name_lower = name.lower()

        if name_lower not in self._registry:
            available = sorted(set(self._registry.keys()))
            raise ValueError(
                f"IMF '{name}' not in registry. "
                f"Available IMFs: {', '.join(available)}"
            )

        return self._registry[name_lower]

    def __getitem__(self, name):
        """
        Dictionary-like access to IMFs.

        Parameters
        ----------
        name : str
            Name of the IMF to load.

        Returns
        -------
        callable
            The IMF function.

        Examples
        --------
        >>> from fastar.imf import imf_registry
        >>> imf_func = imf_registry['kroupa']
        """
        return self.load_by_name(name)

    def __getattr__(self, name):
        """
        Attribute-like access to IMFs.

        Parameters
        ----------
        name : str
            Name of the IMF to load.

        Returns
        -------
        callable
            The IMF function.

        Examples
        --------
        >>> from fastar.imf import imf_registry
        >>> imf_func = imf_registry.kroupa
        """
        if name.startswith('_'):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        try:
            return self.load_by_name(name)
        except ValueError as e:
            raise AttributeError(str(e)) from e

    def register(self, name, imf_func):
        """
        Register a custom IMF function.

        Parameters
        ----------
        name : str
            Name to register the IMF under.
        imf_func : callable
            IMF function with signature `func(mass, params)`.

        Raises
        ------
        ValueError
            If the name already exists and overwrite is False.

        Examples
        --------
        >>> from fastar.imf import imf_registry
        >>> def my_custom_imf(mass, params):
        ...     return mass ** (-2.5)
        >>> imf_registry.register('custom', my_custom_imf)
        >>> custom_imf = imf_registry.load_by_name('custom')
        """
        name_lower = name.lower()

        if name_lower in self._registry:
            raise ValueError(
                f"IMF '{name}' already exists in registry. "
            )

        self._registry[name_lower] = imf_func

    def list_available(self):
        """
        List all available IMF names in the registry.

        Returns
        -------
        list of str
            Sorted list of unique IMF names (excluding aliases).

        Examples
        --------
        >>> from fastar.imf import imf_registry
        >>> imf_registry.list_available()
        ['broken_power_law', 'chabrier', 'flexi', 'kroupa', 'single_power_law']
        """
        return sorted(self._registry.keys())

    def has_imf(self, name):
        """
        Check if an IMF exists in the registry.

        Parameters
        ----------
        name : str
            Name of the IMF to check.

        Returns
        -------
        bool
            True if the IMF exists, False otherwise.

        Examples
        --------
        >>> from fastar.imf import imf_registry
        >>> imf_registry.has_imf('kroupa')
        True
        >>> imf_registry.has_imf('nonexistent')
        False
        """
        return name.lower() in self._registry

    def __repr__(self):
        """String representation of the registry."""
        available = self.list_available()
        return f"IMFRegistry(available={available})"

    def __len__(self):
        """Return the number of unique IMFs in the registry."""
        return len(set(self._registry.values()))
