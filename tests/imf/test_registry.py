"""
Tests for IMF Registry implementation.
"""
import pytest
import jax.numpy as jnp
from fastar.imf import IMFRegistry, imf_registry
from fastar.imf import kroupa, chabrier, single_power_law, broken_power_law, flexi


@pytest.mark.unit
@pytest.mark.imf
class TestIMFRegistry:
    """Test suite for IMF Registry."""

    def test_registry_initialization(self):
        """Test that registry initializes correctly."""
        registry = IMFRegistry()
        assert registry is not None
        assert len(registry) > 0

    def test_load_by_name_method(self, mass_array):
        """Test loading IMF by name using load_by_name method."""
        registry = IMFRegistry()
        kroupa_func = registry.load_by_name('kroupa')
        imf_values = kroupa_func(mass_array, {})
        assert jnp.all(jnp.isfinite(imf_values))
        assert imf_values.shape == mass_array.shape

    def test_dict_like_access(self, mass_array):
        """Test dictionary-like access to IMFs."""
        registry = IMFRegistry()
        chabrier_func = registry['chabrier']
        imf_values = chabrier_func(mass_array, {})
        assert jnp.all(jnp.isfinite(imf_values))

    def test_attribute_access(self, mass_array):
        """Test attribute-like access to IMFs."""
        registry = IMFRegistry()
        flexi_func = registry.flexi
        imf_values = flexi_func(mass_array, {})
        assert jnp.all(jnp.isfinite(imf_values))

    def test_case_insensitive_lookup(self, mass_array):
        """Test that IMF lookup is case insensitive."""
        registry = IMFRegistry()
        kroupa1 = registry.load_by_name('kroupa')
        kroupa2 = registry.load_by_name('KROUPA')
        kroupa3 = registry.load_by_name('Kroupa')

        # All should be the same function
        assert kroupa1 is kroupa2
        assert kroupa2 is kroupa3

    def test_has_imf_method(self):
        """Test has_imf method for checking IMF existence."""
        registry = IMFRegistry()
        assert registry.has_imf('kroupa')
        assert registry.has_imf('chabrier')
        assert not registry.has_imf('nonexistent_imf')

    def test_list_available(self):
        """Test listing available IMFs."""
        registry = IMFRegistry()
        available = registry.list_available()

        assert isinstance(available, list)
        assert len(available) > 0
        assert 'kroupa' in available
        assert 'chabrier' in available
        assert 'single_power_law' in available
        assert 'broken_power_law' in available
        assert 'flexi' in available

    def test_invalid_imf_raises_error(self):
        """Test that requesting invalid IMF raises appropriate error."""
        registry = IMFRegistry()

        with pytest.raises(ValueError, match="not in registry"):
            registry.load_by_name('nonexistent_imf')

        with pytest.raises(AttributeError):
            _ = registry.nonexistent_imf

    def test_register_custom_imf(self, mass_array):
        """Test registering a custom IMF function."""
        registry = IMFRegistry()

        def custom_imf(mass, params):
            alpha = params.get('alpha', 2.5)
            return mass ** (-alpha)

        registry.register('custom', custom_imf)

        assert registry.has_imf('custom')
        custom_func = registry.load_by_name('custom')
        imf_values = custom_func(mass_array, {'alpha': 2.5})
        assert jnp.all(jnp.isfinite(imf_values))

    def test_register_duplicate_raises_error(self):
        """Test that registering duplicate IMF name raises error."""
        registry = IMFRegistry()

        def custom_imf(mass, params):
            return mass ** (-2.5)

        with pytest.raises(ValueError, match="already exists"):
            registry.register('kroupa', custom_imf)

    def test_default_registry_instance(self, mass_array):
        """Test that default registry instance is available."""
        assert imf_registry is not None
        kroupa_func = imf_registry.kroupa
        imf_values = kroupa_func(mass_array, {})
        assert jnp.all(jnp.isfinite(imf_values))

    def test_registry_repr(self):
        """Test string representation of registry."""
        registry = IMFRegistry()
        repr_str = repr(registry)
        assert 'IMFRegistry' in repr_str
        assert 'available' in repr_str

    def test_all_registered_imfs_work(self, mass_array):
        """Test that all pre-registered IMFs are functional."""
        registry = IMFRegistry()
        available = registry.list_available()

        for imf_name in available:
            imf_func = registry.load_by_name(imf_name)
            imf_values = imf_func(mass_array, {})
            assert jnp.all(jnp.isfinite(imf_values)), f"IMF '{imf_name}' failed"
            assert imf_values.shape == mass_array.shape, f"IMF '{imf_name}' shape mismatch"


@pytest.mark.unit
@pytest.mark.imf
class TestDirectImports:
    """Test suite for direct IMF imports."""

    def test_direct_import_kroupa(self, mass_array):
        """Test direct import of kroupa IMF."""
        imf_values = kroupa(mass_array, {})
        assert jnp.all(jnp.isfinite(imf_values))
        assert imf_values.shape == mass_array.shape

    def test_direct_import_chabrier(self, mass_array):
        """Test direct import of chabrier IMF."""
        imf_values = chabrier(mass_array, {})
        assert jnp.all(jnp.isfinite(imf_values))
        assert imf_values.shape == mass_array.shape

    def test_direct_import_single_power_law(self, mass_array):
        """Test direct import of single_power_law IMF."""
        imf_values = single_power_law(mass_array, {})
        assert jnp.all(jnp.isfinite(imf_values))
        assert imf_values.shape == mass_array.shape

    def test_direct_import_broken_power_law(self, mass_array):
        """Test direct import of broken_power_law IMF."""
        imf_values = broken_power_law(mass_array, {})
        assert jnp.all(jnp.isfinite(imf_values))
        assert imf_values.shape == mass_array.shape

    def test_direct_import_flexi(self, mass_array):
        """Test direct import of flexi IMF."""
        imf_values = flexi(mass_array, {})
        assert jnp.all(jnp.isfinite(imf_values))
        assert imf_values.shape == mass_array.shape

    def test_direct_vs_registry_same_function(self):
        """Test that direct imports and registry return the same function."""
        # The direct import should be the same function as from registry
        registry_kroupa = imf_registry.load_by_name('kroupa')
        assert kroupa is registry_kroupa
