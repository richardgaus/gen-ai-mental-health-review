"""
Interrater agreement calculation module.

This module provides functions for calculating interrater agreement metrics
such as Cohen's kappa between reviewers.
"""

from .kappa_calculator import calculate_cohens_kappa, calculate_kappa_for_columns, calculate_overall_agreement, load_kappa_config, generate_kappa_report

__all__ = ['calculate_cohens_kappa', 'calculate_kappa_for_columns', 'calculate_overall_agreement', 'load_kappa_config', 'generate_kappa_report']
