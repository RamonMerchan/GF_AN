"""Post-fault algebraic consistency for ANDES 2.0.0 and the local AN models.

Install after setup() and before TDS.run():
    manager = configure_postfault(ss)

Call manually after TDS.init() and after processing all simultaneous events:
    result = initialize_postfault(ss)

Does not run power flow, init(), or reinit(). Preserves positive-mass states
and solves zero-mass states as algebraic constraints. See
POSFALLA_GENERICA/FORMULACION.md for scope and conventions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Callable

import numpy as np
from scipy.optimize import root, least_squares
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


class PostFaultError(RuntimeError):
    """No acceptable solution was found; input x and y are preserved."""


class UnsupportedModel(PostFaultError):
    """A current adapter is required for a device in the system."""


@dataclass
class PostFaultOptions:
    current_tolerance: float = 1e-8
    algebraic_tolerance: float = 1e-7
    root_tolerance: float = 1e-10
    max_evaluations: int = 4000
    use_fallback: bool = True
    solve_zero_mass_states: bool = True

    def __post_init__(self):
        vals = (self.current_tolerance, self.algebraic_tolerance,
                self.root_tolerance)
        if not all(np.isfinite(v) and v > 0 for v in vals):
            raise ValueError('Tolerances must be positive and finite.')
        if self.max_evaluations < 1:
            raise ValueError('max_evaluations must be positive.')


@dataclass
class PostFaultResult:
    time_s: float
    current_residual_pu: float
    algebraic_residual: float
    zero_mass_residual: float
    max_differential_state_change: float
    minimum_voltage_pu: float
    current_limit_excess_pu: float
    method: str
    seed: str
    evaluations: int
    elapsed_s: float
    zero_mass_states: int
    buses: list
    active_faults: list
    andes_version: str
    attempts: list = field(default_factory=list)
    inverters: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _val(model, name):
    return np.asarray(getattr(model, name).v)


def _status(model, effective=True):
    name = 'ue' if effective and hasattr(model, 'ue') else 'u'
    return _val(model, name).astype(float)


def _uids(ss, buses):
    return np.asarray(ss.Bus.idx2uid(list(buses)), dtype=int)


def _norm(v):
    return float(np.max(np.abs(v))) if np.size(v) else 0.0


def _check_version():
    import andes
    if andes.__version__ != '2.0.0':
        raise UnsupportedModel(
            f'This function requires ANDES 2.0.0; loaded version: {andes.__version__}.')


def _network_components(ss):
    """Stamp the effective Line, Shunt, and Fault equations without double counting.

    ANDES 2.0 Line.build_ybus() omits the 1e-8 term present in Line.e_str.
    Use ghk/bhk and gh/bh from the original evaluator so that the full DAE
    residual can be checked against the network actually solved.
    """
    n = ss.Bus.n
    y = np.zeros((n, n), complex)
    adjacency = np.zeros((n, n), bool)
    line = ss.Line
    if line.n:
        a, b = _uids(ss, line.bus1.v), _uids(ss, line.bus2.v)
        status = _status(line)
        for k in range(line.n):
            if not status[k]:
                continue
            tap = float(line.tap.v[k])
            if tap <= 0:
                raise UnsupportedModel('Line: tap must be positive.')
            t = tap * np.exp(1j * line.phi.v[k])
            ys = complex(line.ghk.v[k], line.bhk.v[k])
            sh1 = complex(line.gh.v[k], line.bh.v[k])
            # Some implementations use gh/bh at the second terminal as well. Follow
            # the exact equation; do not correct the software during the event.
            use_gk = 'gk' in line.a2.e_str
            use_bk = 'bk' in line.v2.e_str
            sh2 = complex(line.gk.v[k] if use_gk else line.gh.v[k],
                          line.bk.v[k] if use_bk else line.bh.v[k])
            aa, bb = a[k], b[k]
            y[aa, aa] += status[k] * (ys + sh1) / tap**2
            y[aa, bb] -= status[k] * ys / np.conj(t)
            y[bb, aa] -= status[k] * ys / t
            y[bb, bb] += status[k] * (ys + sh2)
            if abs(ys) > 0:
                adjacency[aa, bb] = adjacency[bb, aa] = True
    shunt = ss.Shunt
    if shunt.n:
        ids = _uids(ss, shunt.bus.v)
        np.add.at(y, (ids, ids), _status(shunt) * (shunt.g.v + 1j*shunt.b.v))
    fault = ss.Fault
    if fault.n:
        ids = _uids(ss, fault.bus.v)
        np.add.at(y, (ids, ids), _status(fault, False) * fault.uf.v
                  * (fault.gf.v + 1j*fault.bf.v))
    _, labels = connected_components(csr_matrix(adjacency), directed=False)
    return y, labels


def _check_injection_models(ss, extras):
    supported = {'Bus', 'Line', 'Shunt', 'Fault', 'PQ', 'PV', 'Slack',
                 'Playback', 'REGFMA1_AN', 'REGFMB1_AN', 'GENROU', 'GENCLS'}
    for name, model in ss.exist.pflow_tds.items():
        if name in supported or name in extras:
            continue
        # Controllers without network injections retain their original g equations.
        for var in getattr(model, 'algebs_ext', {}).values():
            if getattr(var, 'model', None) in ('Bus', 'ACNode') and var.e_str:
                raise UnsupportedModel(f'{name}: missing current adapter for {var.name}.')
    if np.any(_status(ss.Bus, False) == 0):
        raise UnsupportedModel('Disconnected buses: reduce the system before initialization.')


def _source_buses(ss):
    out = []
    for name in ('Playback', 'REGFMA1_AN', 'REGFMB1_AN', 'GENROU', 'GENCLS'):
        model = ss.models.get(name)
        if model is not None and model.n:
            out.extend(_uids(ss, model.bus.v)[_status(model) != 0])
    return out


def _gfm_current(model, voltage):
    """System-base current; limit and regularization are on the inverter base."""
    if hasattr(model, 'Edroop'):
        angle = _val(model, 'delta')
        emf = _val(model, 'Edroop') * np.exp(1j*angle)
        impedance = 1j * _val(model, 'XL')
    else:
        angle = _val(model, 'delta_VSM')
        emf = _val(model, 'Evsm') * np.exp(1j*angle)
        impedance = _val(model, 'Re') + 1j*_val(model, 'XL')
    if np.any(np.abs(impedance) == 0):
        raise UnsupportedModel('Zero GFM coupling impedance.')
    raw = (emf - voltage) / impedance
    magnitude = np.sqrt(np.abs(raw)**2 + 1e-12)
    scale = np.minimum(1.0, _val(model, 'ImaxF') / magnitude)
    return _status(model, False) * _val(model, 'SnSb') * scale * raw


def _current_balance(ss, ybus, voltage, extras):
    residual = ybus @ voltage
    for name in ('REGFMA1_AN', 'REGFMB1_AN'):
        g = ss.models.get(name)
        if g is not None and g.n:
            ids = _uids(ss, g.bus.v)
            np.add.at(residual, ids, -_gfm_current(g, voltage[ids]))
    for name in ('GENROU', 'GENCLS'):
        g = ss.models.get(name)
        if g is not None and g.n:
            ids = _uids(ss, g.bus.v)
            # ANDES SynGen: vd+jvq = j*V*exp(-j*delta). Id/Iq are already
            # on the system base; do not multiply by Sn/Sbase again.
            injected = _status(g) * (g.Iq.v - 1j*g.Id.v) * np.exp(1j*g.delta.v)
            np.add.at(residual, ids, -injected)
    source = ss.models.get('Playback')
    if source is not None and source.n:
        ids = _uids(ss, source.bus.v)
        z = source.R.v + 1j*source.X.v
        if np.any(np.abs(z) == 0):
            raise UnsupportedModel('Zero-impedance Playback requires an ideal-voltage boundary condition.')
        emf = source.E.v * np.exp(1j*source.delta.v)
        np.add.at(residual, ids, -_status(source, False)*(emf-voltage[ids])/z)
    pq = ss.PQ
    if pq.n:
        ids = _uids(ss, pq.bus.v)
        v = voltage[ids]
        mag = np.abs(v)
        c = pq.config
        # Z requires no division by V; I and P require nonzero V.
        sp = c.p2p*pq.Ppf.v + 1j*c.q2q*pq.Qpf.v
        si = c.p2i*pq.Ipeq.v + 1j*c.q2i*pq.Iqeq.v
        sz = c.p2z*pq.Req.v - 1j*c.q2z*pq.Xeq.v
        active = _status(pq)
        unsafe = (mag < 1e-12) & (active != 0) & ((np.abs(sp)+np.abs(si)) > 0)
        if np.any(unsafe):
            raise PostFaultError('P/I load at zero voltage: use another initial guess without changing the load.')
        cp = np.divide(np.conj(sp), np.conj(v), out=np.zeros_like(v), where=mag > 0)
        ci = np.divide(np.conj(si)*v, mag, out=np.zeros_like(v), where=mag > 0)
        np.add.at(residual, ids, active*(sz*v + cp + ci))
    # Unreplaced static generators retain their original constraints in g;
    # convert their injections to current for KCL.
    for name in ('PV', 'Slack'):
        g = ss.models.get(name)
        if g is None or not g.n:
            continue
        active = _status(g) != 0
        if not np.any(active):
            continue
        ids = _uids(ss, g.bus.v)[active]
        v = voltage[ids]
        if np.any(np.abs(v) < 1e-12):
            raise PostFaultError('Active static generation at zero voltage.')
        s = g.p.v[active] + 1j*g.q.v[active]
        np.add.at(residual, ids, -np.conj(s/v))
    for callback in extras.values():
        # Extension convention: extra current is positive for consumption.
        extra = np.asarray(callback(ss, voltage), dtype=complex)
        if extra.shape != voltage.shape:
            raise ValueError('An extra adapter must return one current per bus.')
        residual += extra
    return residual


def _snapshot_discretes(models):
    saved = []
    for model in models.values():
        for obj in model.discrete.values():
            for name, value in vars(obj).items():
                if isinstance(value, np.ndarray):
                    saved.append((obj, name, value.copy()))
    return saved


def _restore_discretes(saved):
    for obj, name, value in saved:
        current = getattr(obj, name)
        if current.shape == value.shape:
            current[:] = value
        else:
            setattr(obj, name, value.copy())


def initialize_postfault(ss, *, options=None, seeds=None,
                         current_adapters=None):
    """Solve KCL and all original g equations at fixed time and fixed dynamic x.

    ``seeds``: optional sequence of (name, copy_of_dae_y) pairs.
    ``current_adapters``: model_name -> callable(ss, V) dictionary returning
    system-base currents for all buses, positive for consumption. ANDES
    still evaluates the corresponding internal algebraic equations.
    On failure, restore input x, y, f, g, and discrete indicators.
    """
    started = perf_counter()
    _check_version()
    o = options or PostFaultOptions()
    extras = current_adapters or {}
    if not ss.TDS.initialized:
        raise PostFaultError('Run TDS.init() before manual initialization.')
    _check_injection_models(ss, extras)
    ybus, labels = _network_components(ss)
    # Do not impose an artificial angle reference: preserved dynamic sources
    # set the phase of each energized connected component.
    referenced = {labels[i] for i in _source_buses(ss)}
    for name in ('PV', 'Slack'):
        model = ss.models.get(name)
        if model is not None and model.n:
            referenced.update(labels[i] for i in _uids(ss, model.bus.v)[_status(model) != 0])
    missing = set(labels) - referenced
    if missing:
        buses = [b for b, label in zip(ss.Bus.idx.v, labels) if label in missing]
        raise UnsupportedModel(f'Island without a supported source: buses {buses}.')
    x0, y0, f0, g0 = (v.copy() for v in (ss.dae.x, ss.dae.y, ss.dae.f, ss.dae.g))
    models = ss.exist.pflow_tds
    discrete = _snapshot_discretes(models)
    za = np.flatnonzero(np.asarray(ss.dae.Tf) == 0)
    if za.size and not o.solve_zero_mass_states:
        raise UnsupportedModel('Zero-mass states are present; enable solve_zero_mass_states.')
    dynamic = np.flatnonzero(np.asarray(ss.dae.Tf) != 0)
    aa, va = ss.Bus.a.a, ss.Bus.v.a
    fixed = []
    for name in ('PV', 'Slack'):
        model = ss.models.get(name)
        if model is not None and model.n:
            inactive = _status(model) == 0
            for var in model.algebs.values():
                for address in np.asarray(var.a)[inactive]:
                    # ANDES can reuse Slack/PV power-flow addresses for dynamic
                    # variables. Do not fix those stale addresses.
                    label = str(ss.dae.y_name[address])
                    expected = f'{var.name} {name} '
                    if not label or label.startswith(expected):
                        fixed.append(address)
    fixed = np.asarray(fixed, dtype=int)
    evaluations = 0

    def pack(y):
        if np.asarray(y).shape != y0.shape:
            raise ValueError('The initial guess must have the same shape as dae.y.')
        z = np.r_[y, x0[za]].astype(float)
        v = y[va]*np.exp(1j*y[aa])
        z[aa], z[va] = v.real, v.imag
        return z

    def evaluate(z, original=False):
        nonlocal evaluations
        evaluations += 1
        voltage = z[aa] + 1j*z[va]
        ss.dae.x[:] = x0
        ss.dae.x[za] = z[len(y0):]
        ss.dae.y[:] = z[:len(y0)]
        ss.dae.y[aa] = np.angle(voltage)
        ss.dae.y[va] = np.abs(voltage)
        ss.vars_to_models()
        ss.dae.clear_fg()
        ss.s_update_var(models)
        ss.l_update_var(models, niter=0, err=0.0)
        ss.f_update(models)
        ss.g_update(models)
        # fg_to_dae() can project states using a previous anti-windup x_set.
        # Collect equations only; do not apply projections to x.
        ss._e_to_dae(('f', 'g'))
        current = _current_balance(ss, ybus, voltage, extras)
        residual = ss.dae.g.copy()
        residual[aa], residual[va] = current.real, current.imag
        residual[fixed] = ss.dae.y[fixed] - y0[fixed]
        if not np.array_equal(ss.dae.x[dynamic], x0[dynamic]):
            raise PostFaultError('An evaluator changed differential states during the search.')
        if original:
            return current, ss.dae.g.copy(), ss.dae.f[za].copy()
        return np.r_[residual, ss.dae.f[za]]

    candidates = [('current', y0)] + list(seeds or [])
    flat = y0.copy()
    flat[va] = 1.0
    candidates.append(('unit_voltage', flat))
    seeds = []
    for name, value in candidates:
        if not any(np.array_equal(value, old) for _, old in seeds):
            seeds.append((name, value))
    attempts = []
    try:
        for name, yseed in seeds:
            z = pack(np.asarray(yseed))
            methods = ['hybr'] + (['trf'] if o.use_fallback else [])
            for method in methods:
                try:
                    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                        if method == 'hybr':
                            sol = root(evaluate, z, method='hybr', tol=o.root_tolerance,
                                       options={'maxfev': o.max_evaluations})
                        else:
                            sol = least_squares(evaluate, z, method='trf',
                                ftol=o.root_tolerance, xtol=o.root_tolerance,
                                gtol=o.root_tolerance, max_nfev=min(o.max_evaluations, 150),
                                x_scale='jac')
                    if not np.isfinite(sol.x).all():
                        raise PostFaultError('The solver produced nonfinite values.')
                    current, algebraic, zero = evaluate(sol.x, original=True)
                    ci, ga, gz = _norm(current), _norm(algebraic), _norm(zero)
                    if not np.isfinite(np.r_[current, algebraic, zero]).all():
                        raise PostFaultError('Nonfinite residuals.')
                    attempts.append(dict(seed=name, method=method, KCL=ci, g=ga,
                                         zero_mass=gz, solver_success=bool(sol.success),
                                         largest_residual_equation=str(ss.dae.y_name[int(np.argmax(np.abs(algebraic)))])))
                    if ci > o.current_tolerance or max(ga, gz) > o.algebraic_tolerance:
                        if np.isfinite(sol.x).all():
                            z = sol.x
                        continue
                    excess = 0.0
                    inverter_report = []
                    for mn in ('REGFMA1_AN', 'REGFMB1_AN'):
                        g = ss.models.get(mn)
                        if g is not None and g.n:
                            actual = np.hypot(g.Id.v, g.Iq.v)
                            active = _status(g, False) != 0
                            excess = max(excess, _norm(np.maximum(0, actual[active]-g.ImaxF.v[active])))
                            for i in range(g.n):
                                inverter_report.append(dict(model=mn, idx=str(g.idx.v[i]),
                                    bus=str(g.bus.v[i]), connected=bool(active[i]),
                                    Sn_over_Sbase=float(g.SnSb.v[i]),
                                    current_pu_device=float(actual[i]),
                                    limit_pu_device=float(g.ImaxF.v[i]),
                                    limiting_factor=float(g.Iscale.v[i])))
                    if excess > o.algebraic_tolerance:
                        continue
                    # Preserve integrator limiter memory. Output flags have been
                    # reevaluated; do not execute stale x_set entries.
                    for limiter in ss.antiwindups:
                        limiter.x_set = []
                    import andes
                    return PostFaultResult(float(ss.dae.t), ci, ga, gz,
                        _norm(ss.dae.x[dynamic]-x0[dynamic]),
                        float(np.min(ss.dae.y[va])), excess, method, name,
                        evaluations, perf_counter()-started, len(za), list(ss.Bus.idx.v),
                        list(map(float, ss.Fault.uf.v)), andes.__version__, attempts, inverter_report)
                except (PostFaultError, ValueError, FloatingPointError) as exc:
                    attempts.append(dict(seed=name, method=method, error=str(exc)))
        raise PostFaultError(f'Post-fault initialization did not converge: {attempts}')
    except Exception:
        ss.dae.x[:], ss.dae.y[:] = x0, y0
        ss.vars_to_models()
        ss.s_update_var(models)
        _restore_discretes(discrete)
        ss.dae.f[:], ss.dae.g[:] = f0, g0
        raise


class PostFaultManager:
    """Run one solution after all events at the current instant."""

    def __init__(self, ss, *, events='clear', options=None, on_result=None):
        _check_version()
        if events not in ('clear', 'both'):
            raise ValueError("events must be 'clear' or 'both'.")
        if getattr(ss, '_generic_postfault_manager', None) is not None:
            raise ValueError('A post-fault manager is already installed.')
        self.ss, self.events = ss, events
        self.options = options or PostFaultOptions()
        self.on_result = on_result
        self.results = []
        self._switch = ss.switch_action
        self._restore = ss.Fault.config.restore
        self._prefault = None
        ss.Fault.config.restore = 0
        ss.switch_action = self._process_events
        ss._generic_postfault_manager = self

    def _process_events(self, models):
        ss = self.ss
        before = ss.Fault.uf.v.copy()
        ybefore = ss.dae.y.copy()
        result = self._switch(models)
        after = ss.Fault.uf.v
        applied = bool(np.any((before == 0) & (after != 0)))
        cleared = bool(np.any((before != 0) & (after == 0)))
        if applied:
            self._prefault = ybefore
        if cleared or (applied and self.events == 'both'):
            ss.vars_to_models()
            seeds = [('before_event', ybefore)]
            if self._prefault is not None:
                seeds.append(('prefault', self._prefault))
            record = initialize_postfault(ss, options=self.options, seeds=seeds)
            self.results.append(record)
            if self.on_result:
                self.on_result(record)
        return result

    def uninstall(self):
        """Restore the previous event handler and Fault option."""
        if self.ss._generic_postfault_manager is self:
            self.ss.switch_action = self._switch
            self.ss.Fault.config.restore = self._restore
            self.ss._generic_postfault_manager = None


def configure_postfault(ss, *, events='clear', options=None,
                       on_result: Callable | None = None):
    """Install the post-event calculation without replacing Fault callbacks.

    Install after setup() and before TDS.run(). Remove legacy semilla_red.py
    wrappers first; do not combine the managers. Simultaneous trips are
    processed before the calculation, inside switch_action().
    """
    return PostFaultManager(ss, events=events, options=options,
                         on_result=on_result)
