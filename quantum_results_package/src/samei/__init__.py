"""samei: Same-i sector resource theory toolkit.

Conventions (mandatory across the package):
    A     = J_A tensor I_B
    B     = I_A tensor J_B
    M     = A B = J_A tensor J_B
    H_+   = ker(M + I)     aligned sector,  M = -I
    H_-   = ker(M - I)     leakage sector,   M = +I
    P_+   = (I - M) / 2
    P_-   = (I + M) / 2
    L(rho) = Tr(P_- rho)
    C(rho) = || P_+ rho P_- ||_1
"""

from . import (
    linear_algebra,
    realification,
    sectors,
    leakage,
    channels,
    dynamics,
    discrimination,
    cfr,
    moment_matrix,
)

from .linear_algebra import (
    canonical_J,
    trace_norm,
    random_real_density,
    random_complex_density,
    random_effect_complex,
    orthonormal_basis_for_projector,
    projectors_from_involution,
)
from .realification import R, realify_state, realify_effect, born_check
from .sectors import (
    sector_operators,
    projectors,
    iota_simple,
    induced_complex_structure,
)
from .leakage import (
    leakage,
    coherent_leakage,
    coherent_envelope_violation,
    coherent_pure_state,
    block_diagonal_state,
)
from .channels import (
    apply_kraus,
    kraus_dual,
    is_trace_preserving,
    is_leakage_nonincreasing,
    kraus_no_leakage,
    random_sector_block_channel,
    violating_channel_example,
)
from .dynamics import (
    skew_commuting_with_M,
    skew_cross_sector,
    leakage_perturbative_curve,
    duhamel_leading_coefficient,
)
from .discrimination import (
    optimal_psucc_at_budget,
    sector_parity_psucc,
    saturating_state,
)
from .cfr import cfr_state, cfr_leakage_closed_form, cfr_M, cfr_projectors
from .moment_matrix import build_moment_matrix, default_word_set
