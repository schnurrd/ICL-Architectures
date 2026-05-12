"""TabPFN v1 Prior - Standalone TabPFN Prior Generation Package

This package provides a standalone implementation of TabPFN's prior generation
capabilities, extracted and adapted for easy integration into other projects.

Features:
- Multiple prior types: MLP, GP, GP Mix, Prior Bag
- Flexible categorical encoding
- Differentiable hyperparameters
- Compatible with tabularpriors interface
"""

from .tabpfn_prior import TabPFNPriorDataLoader, build_tabpfn_prior

__version__ = "1.0.0"
__all__ = ['TabPFNPriorDataLoader', 'build_tabpfn_prior']