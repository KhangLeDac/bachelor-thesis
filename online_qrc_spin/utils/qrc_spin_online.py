"""Online (measurement-based) spin quantum reservoir.

The reservoir and the input encoding follow ``SpinQRC`` in ``qrc_spin.py``: the same
Hamiltonians ("TiltedTFIM", "TFIM"), the same field encodings and, for a given seed, the
same coupling matrix J (by factor 2). What differs is the protocol. ``SpinQRC`` uses the restarting
protocol: rho evolves untouched and is read out exactly. Here the reservoir keeps running
while a subset of spins (the probes) is measured after every input step, and the
measurement acts back on the state.

One input step k:

    1. evolve       rho -> exp(L(s_k; gamma_k) tau) rho
                    L contains H(s_k) and amplitude damping with rate gamma_k
    2. read out     Z-basis outcome distribution of the probes -> 2^M - 1 Z-string features
    3. back-action  for each probe: Kraus operators {P0, R_y(theta_k) P1}
                    theta = 0 pure dephasing, theta = pi full reset to |0>, else partial reset

This is the unconditional dynamics of the S -> infinity ensemble of copies. With finite
``n_shots`` only the features are sampled; the state still follows the unconditional map.

Policies supply (gamma_k, theta_k) per step. Policy A is ``ConstantPolicy``; feedforward
and feedback policies (B, C) only need a new class with the same ``__call__`` signature.

Reproduction of the system/Hamiltonian described in the paper Sannia et al. : Dissipation as a resource for QRC: use ``model="TiltedTFIM"``, ``encoding_method="uniform"``,
For full projection of the probes to |0>: theta = pi
Right now gamma_k is the same for all spins (no per spin rates)



Conventions
-----------
* Basis index 0 is |0> with Z|0> = +|0>, as in qutip. Qubit 0 is the leftmost tensor factor.
* density matrices are qutip.Qobj with shape (2^n, 2^n) to use qutip's mesolve and tensor functions.
* ``decay_to`` sets the fixed point of amplitude damping. ``SpinQRC`` uses qt.sigmam(),
  which maps index 0 -> 1, i.e. it decays to |1> (Z = -1). The default here is 0, the
  hardware convention, so damping and reset share the fixed point |0>. Use
  ``decay_to=1`` to reproduce ``SpinQRC``.
"""

from itertools import combinations

import numpy as np
import qutip as qt


class ConstantPolicy:
    """Policy A: the same (gamma, theta) at every step."""

    def __init__(self, gamma: float, theta: float):
        self.gamma = float(gamma)
        self.theta = float(theta)

    def __call__(self, k: int, s_k: np.ndarray, prev_features: np.ndarray | None) -> tuple[float, float]:
        """Return (gamma_k, theta_k) for step k.

        Args:
            k: step index.
            s_k: input at step k (feedforward policies use this).
            prev_features: features of step k - 1, None at k = 0 (feedback policies use this).
        """
        return self.gamma, self.theta


class OnlineSpinQRC:
    def __init__(
        self,
        n: int,
        encoding_strength: float, # not in the paper Sannia et al.
        coupling_strength: float, # J_s
        gamma: float,
        theta: float,
        dt: float,
        probes: int | list[int] = 3,
        model: str = "TiltedTFIM",
        encoding_method: str = "uniform",
        damping_on: str = "all", #? which spins are damped
        decay_to: int = 0, #?
        seed: int = 42,
        n_shots: float = float("inf"),
        shot_seed: int | None = None,
    ):
        """
        Args:
            n: number of spins.
            encoding_strength: strength of the input encoding into the fields.
            coupling_strength: J_ij uniform in [-J_s, J_s].
            gamma: amplitude-damping rate of the default constant policy.
            theta: reset angle of the default constant policy (0 = dephasing only, pi = full reset).
            dt: duration tau of one input step.
            probes: number M of measured spins (then spins 0..M-1) or an explicit list.
            model: "TiltedTFIM" (input on the X fields, constant Z field) or "TFIM"
                (input on the Z fields), both with XX couplings, as in SpinQRC.
            encoding_method: "one_to_one", "fill", "dense"as in SpinQRC, or "uniform" for the Sannia et al. paper.
            damping_on: "all" spins or only the "probes".
            decay_to: fixed point of amplitude damping, 0 or 1 (see module docstring).
            seed: seed for J and the encoding weights; same draws as SpinQRC(scaled by 2).
            n_shots: shots per step for the features; inf gives exact probabilities.
            shot_seed: seed of the shot sampler, defaults to seed.
        """
        if model not in ("TiltedTFIM", "TFIM"):
            raise ValueError(f"Unknown model: {model}")
        if encoding_method not in ("one_to_one", "fill", "dense", "uniform"):
            raise ValueError(f"Unknown encoding_method: {encoding_method}")
        if damping_on not in ("all", "probes"):
            raise ValueError(f"damping_on must be 'all' or 'probes', got {damping_on!r}")
        if decay_to not in (0, 1):
            raise ValueError(f"decay_to must be 0 or 1, got {decay_to!r}")

        self.n = n
        self.dim = 2 ** n
        self.encoding_strength = encoding_strength
        self.coupling_strength = coupling_strength
        self.dt = dt
        self.model = model
        self.encoding_method = encoding_method
        self.damping_on = damping_on
        self.decay_to = decay_to
        self.seed = seed
        self.constant_field = 1.0 # constant magnetic field h
        self.policy = ConstantPolicy(gamma, theta)

        """Probes"""
        self.probes = list(range(probes)) if isinstance(probes, (int, np.integer)) else sorted(probes)
        if any(q < 0 or q >= n for q in self.probes):
            raise ValueError(f"Probe indices must be in [0, {n}), got {self.probes}")
        self.n_probes = len(self.probes)

        """Initialize basic operators"""
        # Cache single-site operators to avoid calling qt.tensor in loops
        self.sx_ops = [self._build_operator(qt.sigmax(), i) for i in range(n)]
        self.sz_ops = [self._build_operator(qt.sigmaz(), i) for i in range(n)]
        # Probe operators for the measurement back-action
        self.probe_P0_ops = [self._build_operator(qt.fock_dm(2, 0), q) for q in self.probes] # |0><0|
        self.probe_P1_ops = [self._build_operator(qt.fock_dm(2, 1), q) for q in self.probes] # |1><1|
        self.probe_E01_ops = [self._build_operator(qt.sigmap(), q) for q in self.probes]  # |0><1|

        """Initialize the random number generator"""
        # Random draws in the same order as SpinQRC by factor 2 (seed, then noise_fill if "fill", then J,
        # then Win lazily at the first input), but from a local generator instead of the
        # global np.random state. RandomState(seed) gives the same numbers as np.random.seed.
        self._rng = np.random.RandomState(seed)
        self.noise_fill = self._rng.uniform(0, 1, n) if encoding_method == "fill" else None
        self.J = self._rng.uniform(-coupling_strength, coupling_strength, size=(n, n))
        self.Win = None

        self.n_shots = n_shots
        self._finite_shots = not np.isinf(n_shots)
        if self._finite_shots and (n_shots < 1 or int(n_shots) != n_shots):
            raise ValueError(f"n_shots must be a positive integer or inf, got {n_shots}")
        self._shot_rng = np.random.default_rng(seed if shot_seed is None else shot_seed)

        self.initial_state = self._get_initial_state()
        # Build the parts of H that do not change with input data
        self.H_static = self._build_static_hamiltonian()
        self._build_lowering_operators()
        self._build_feature_map()

    # ------------------------------------------------------------------ public

    def run(
        self,
        data: np.ndarray,
        rho0: qt.Qobj | None = None,
        policy=None,
        return_states: bool = False,
    ):
        """Drive the reservoir with an input sequence.

        Args:
            data: inputs of shape (steps, data_dim) or (steps,).
            rho0: initial density matrix (dim x dim); defaults to |0...0><0...0|.
            policy: callable (k, s_k, prev_features) -> (gamma_k, theta_k);
                defaults to the constant policy from the constructor.
            return_states: if True, also return the state after each step's back-action.

        Returns:
            features of shape (steps, 2^M - 1), and the states if requested.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data[:, None]
        policy = self.policy if policy is None else policy
        rho = self.initial_state if rho0 is None else qt.Qobj(rho0, dims=self.initial_state.dims)

        # implement policy and back-action in a loop, since the state is not reset between steps
        features = np.zeros((data.shape[0], self.n_features))
        states = [] if return_states else None
        prev = None
        for k, s_k in enumerate(data):
            gamma_k, theta_k = policy(k, s_k, prev)
            features[k], rho = self.step(rho, s_k, gamma_k, theta_k)
            prev = features[k]
            if return_states:
                states.append(rho.copy())

        if return_states:
            return features, states
        return features

    def step(self, rho: qt.Qobj, s_k: np.ndarray, gamma_k: float, theta_k: float) -> tuple[np.ndarray, qt.Qobj]:
        """One input step: evolve, read the probes, apply the back-action.

        The features are read before the reset; afterwards the probes are biased toward |0>.

        Returns:
            (features, rho after back-action)
        """
        rho = self.evolve(rho, s_k, gamma_k)
        features = self.features(rho)
        rho = self.measure_and_reset(rho, theta_k)
        return features, rho

    def evolve(self, rho: qt.Qobj, s_k: np.ndarray, gamma_k: float) -> qt.Qobj:
        """Lindblad evolution over one interval tau with the input-dependent Hamiltonian."""
        # Total field per spin (constant part included), encoded from the input
        field = self.field_strengths(s_k)
        # H = H_static + H_dynamic. H_static = sum_i h Z_i + sum_{i<j} J_ij X_i X_j is built once
        # in __init__; H_dynamic carries the input and depends on the model.
        h = self.constant_field
        H_dynamic = 0
        if self.model == "TiltedTFIM":
            # TiltedTFIM: the input drives the X fields (total X coefficient = field_k)
            for k in range(self.n):
                if field[k] != 0: # Optimization: skip zero fields
                    H_dynamic += field[k] * self.sx_ops[k]
            H = self.H_static + H_dynamic
                    
        elif self.model == "TFIM":
            # TFIM: the input drives the Z fields. H_static already holds h * Z_i, so only the
            # difference is added and the total Z coefficient is field_i.
            for i in range(self.n):
                H_dynamic += (field[i] - h) * self.sz_ops[i]
            H = self.H_static + H_dynamic
        else:
            raise ValueError(f"Unknown model: {self.model}")

        # Evolve
        # gamma_k is the same for every damped spin; each keeps its own jump operator
        # sqrt(gamma_k) * lowering_i (summing them would give collective decay)
        c_ops = [np.sqrt(gamma_k) * L for L in self.lowering_ops]
        result = qt.mesolve(H, rho, tlist=[0, self.dt], c_ops=c_ops, options={"nsteps": 50000})
        rho = result.states[-1]
        return rho

    def measure_and_reset(self, rho: qt.Qobj, theta: float) -> qt.Qobj:
        """Unconditional back-action on every probe: rho -> K0 rho K0^+ + K1 rho K1^+.

        K0 = P0 (outcome 0, left as is), K1 = R_y(theta) P1 (outcome 1, rotated toward |0>).
        R_y(theta) P1 = -sin(theta/2) |0><1| + cos(theta/2) |1><1|. Completeness:
        K0^+ K0 + K1^+ K1 = P0 + P1 = 1.
        """
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        for P0, P1, E01 in zip(self.probe_P0_ops, self.probe_P1_ops, self.probe_E01_ops):
            # Apply Kraus operators to every probe
            K0 = P0 # |0><0|
            K1 = c * P1 - s * E01 # cos(theta/2) |1><1| - sin(theta/2) |0><1|
            rho = K0 @ rho @ K0 + K1 @ rho @ K1.dag()
        return 0.5 * (rho + rho.dag()) # why that?

    def outcome_probabilities(self, rho: qt.Qobj) -> np.ndarray:
        """Joint Z-basis outcome distribution of the probes, shape (2^M,).

        Index b in binary gives the probe bits in the order of self.probes, first probe most
        significant.
        """
        p = np.real(rho.diag()).reshape((2,) * self.n)
        others = tuple(q for q in range(self.n) if q not in self.probes)
        p = p.sum(axis=others) if others else p
        p = np.clip(p.reshape(-1), 0.0, None)
        return p / p.sum()

    def features(self, rho: qt.Qobj) -> np.ndarray:
        """All 2^M - 1 Z-strings of the probes (exact, or estimated from n_shots outcomes)."""
        p = self.outcome_probabilities(rho)
        if self._finite_shots:
            p = self._shot_rng.multinomial(int(self.n_shots), p) / self.n_shots
        return self._z_signs @ p

    def field_strengths(self, data_step: np.ndarray) -> np.ndarray:
        """Per-spin field from one input step; identical to SpinQRC._generate_field_strengths."""
        data_step = np.atleast_1d(np.asarray(data_step, dtype=float))
        h = self.constant_field

        field = np.full(self.n, h)

        # fields from the original SpinQRC, designed for a different system
        if self.encoding_method == "one_to_one":
            for i, dat in enumerate(data_step):
                field[i] = h + 2 * self.encoding_strength * (dat - 0.5)
        elif self.encoding_method == "fill":
            per_dim = self.n // len(data_step)
            for i, dat in enumerate(data_step):
                a, b = i * per_dim, (i + 1) * per_dim
                field[a:b] = h + 2 * self.encoding_strength * (dat - 0.5) * self.noise_fill[a:b]
        elif self.encoding_method == "dense":
            if self.Win is None:
                Win = self._rng.uniform(-1, 1, (self.n, len(data_step)))
                row_sums = np.sum(np.abs(Win), axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                self.Win = Win / row_sums
            field = h + self.encoding_strength * (self.Win @ (2 * (data_step - 0.5)))

        # field from the paper Sannia et al. : Dissipation as a resourse for QRC:
        # input s_k is scalar
        elif self.encoding_method == "uniform":
            if data_step.size != 1:
                raise ValueError(
                    f"encoding_method='uniform' needs a scalar input per step, got shape {data_step.shape}"
                )
            #h' = h (1 + s_k), the same on every spin
            field = np.full(self.n, h * (1 + data_step[0]))

        
        return field

    # ----------------------------------------------------------------- builders

    def _get_initial_state(self) -> qt.Qobj:
        return qt.tensor([qt.fock_dm(2, 0)] * self.n)  # |0...0><0...0|

    def _build_operator(self, operator: qt.Qobj, i: int) -> qt.Qobj:
        """Places an operator at some site in an n-qubit system.

        This function builds the tensor product of an operator acting on the i-th spin
        in a n-spin system with identity operators acting on all of the other spins.

        Args:
            operator: operator to be placed,
            i (int): site of the operator,

        Returns:
            tensor product of operators."""
        ops = [qt.identity(2)] * self.n
        ops[i] = operator
        return qt.tensor(ops)

    def _build_static_hamiltonian(self) -> qt.Qobj:
        """Constructs the constant parts of the Hamiltonian once."""
        H_static = 0
        h = self.constant_field
        
        # Constant Transverse/Longitudinal fields (Base Hamiltonian)
        for i in range(self.n):
            H_static += h * self.sz_ops[i]

        # Coupling Terms (Always static)
        # J is symmetric, iterate upper triangle to save time
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.J[i, j] != 0:
                    # Use precomputed operators
                    # Op(i,j) = sx_ops[i] * sx_ops[j] (Matrix multiplication of full operators)
                    H_static += self.J[i, j] * (self.sx_ops[i] * self.sx_ops[j])
                    
        return H_static

    def _build_feature_map(self):
        """Signs of every Z-string on outcome b: singles, then pairs, ..., then the full product."""
        M = self.n_probes
        bits = (np.arange(2 ** M)[:, None] >> np.arange(M - 1, -1, -1)) & 1  # (2^M, M)
        signs, labels = [], []
        for size in range(1, M + 1):
            for subset in combinations(range(M), size):
                signs.append(np.prod(1 - 2 * bits[:, list(subset)], axis=1))
                labels.append("".join(f"Z{self.probes[a]}" for a in subset))
        self._z_signs = np.array(signs, dtype=float)
        self.feature_labels = labels
        self.n_features = len(labels)

    def _build_lowering_operators(self):
        """Unit-rate lowering operators; sqrt(gamma_k) is applied per step."""
        lowering = qt.sigmam() if self.decay_to == 1 else qt.sigmap()  # |1><0| decays to |1>, |0><1| to |0>
        damped = range(self.n) if self.damping_on == "all" else self.probes
        self.lowering_ops = [self._build_operator(lowering, i) for i in damped]






def trace_distance(rho: qt.Qobj, sigma: qt.Qobj) -> float:
    """D(rho, sigma) = 1/2 ||rho - sigma||_1, for the ESP test and contraction estimates."""
    return 0.5 * float((rho - sigma).norm("tr"))
