"""Same-$i$ Leakage reference implementation.

Companion library for the manuscript on the resource theory of shared complex
structure in real quantum composition.

Public API:
    core:           canonical_J, realify_operator, realify_state, sector_operators
    leakage:        leakage, coherent_leakage, alignment_weight, trace_distance_to_free
    channels:       is_leakage_nonincreasing, kraus_no_leakage, apply_kraus
    dynamics:       schrodinger_generator, evolve_hamiltonian, integrate_lindblad,
                    lindblad_generator_dual_on
    discrimination: sector_parity_psucc, optimal_psucc_at_budget
    examples:       cfr_state, cfr_leakage_closed_form, bob_node_leakage_product,
                    imaginarity_inequivalence_A/B
"""

from .core import (
    canonical_J,
    realify_operator,
    realify_state,
    sector_operators,
    projectors,
    induced_complex_structure,
    is_orthogonal_complex_structure,
)
from .leakage import (
    leakage,
    coherent_leakage,
    alignment_weight,
    trace_distance_to_free,
    is_in_free_face,
)
from .channels import (
    apply_kraus,
    kraus_dual,
    is_trace_preserving,
    is_leakage_nonincreasing,
    kraus_no_leakage,
    is_strongly_sector_preserving,
)
from .dynamics import (
    schrodinger_generator,
    evolve_hamiltonian,
    lindblad_generator_dual_on,
    integrate_lindblad,
)
from .discrimination import (
    sector_parity_psucc,
    optimal_psucc_at_budget,
    saturating_state,
)
from .examples import (
    cfr_state,
    cfr_leakage_closed_form,
    bob_node_leakage_product,
    imaginarity_inequivalence_A,
    imaginarity_inequivalence_B,
    embed_complex_vector_to_H_plus,
    haar_real_state,
    random_real_density,
)

__all__ = [
    "canonical_J",
    "realify_operator",
    "realify_state",
    "sector_operators",
    "projectors",
    "induced_complex_structure",
    "is_orthogonal_complex_structure",
    "leakage",
    "coherent_leakage",
    "alignment_weight",
    "trace_distance_to_free",
    "is_in_free_face",
    "apply_kraus",
    "kraus_dual",
    "is_trace_preserving",
    "is_leakage_nonincreasing",
    "kraus_no_leakage",
    "is_strongly_sector_preserving",
    "schrodinger_generator",
    "evolve_hamiltonian",
    "lindblad_generator_dual_on",
    "integrate_lindblad",
    "sector_parity_psucc",
    "optimal_psucc_at_budget",
    "saturating_state",
    "cfr_state",
    "cfr_leakage_closed_form",
    "bob_node_leakage_product",
    "imaginarity_inequivalence_A",
    "imaginarity_inequivalence_B",
    "embed_complex_vector_to_H_plus",
    "haar_real_state",
    "random_real_density",
]
