"""Positive-sequence REGFM_B1_AN Virtual Synchronous Machine (VSM) Grid-Forming Inverter for ANDES.

Model designation: REGFMB1_AN (Grid-Forming Inverter adapted for ANDES).
Differentiated from PSS/E standard 'REGFMB1U' model.

Specification based on:
  - UNIFI / NREL / PNNL / EPRI / WECC:
    "Virtual Synchronous Machine Grid-Forming Inverter Model Specification (REGFM_B1)", 2024.
  - Siemens PSS/E v36.2.0 Model Documentation, Section 21.8: REGFMB1U.

Main features:
  - Virtual Synchronous Machine swing equation (2H, steady-state damping D1,
    transient damping D2 via washout filter with frequency wD).
  - P-f active power-frequency droop with measurement filter (mp, Tp).
  - Grid-following PLL with low-voltage freeze logic (VPLLfrz).
  - Angle decomposition: delta_VSM = delta_PLL + delta_IT, with active current
    limiting on delta_IT via dynamic upper/lower integrators (delta_ITmax, delta_ITmin).
  - PQ priority algorithm (PQFlag, kf) determining steady-state current limits (IdmaxSS, IqmaxSS).
  - Q-V reactive power/current droop with virtual impedance option (VdrpFlag, mq).
  - Terminal voltage PI controller with dynamic internal voltage limits (Emax, Emin).
  - Complex coupling impedance (Re + j XL).
  - Circular algebraic transient current limiter (ImaxF).
  - Zero-mismatch bumpless initialization from power flow solution.
"""

import numpy as np
from andes.core import (Algeb, AntiWindup, ConstService, ExtAlgeb, ExtService, IdxParam,
                        Lag, LessThan, Model, ModelData, NumParam, Switcher)
from andes.core.block import GainLimiter
from andes.core.var import State


class MovingBoundAntiWindup(AntiWindup):
    """Anti-windup que también proyecta el estado si un límite móvil lo rebasa."""

    def check_eq(self, allow_adjust=True, adjust_lower=False, adjust_upper=False,
                 is_init=False, niter=0, **kwargs):
        upper_v = -self.upper.v if self.sign_upper.v == -1 else self.upper.v
        lower_v = -self.lower.v if self.sign_lower.v == -1 else self.lower.v

        self.zu0[:] = self.zu
        self.zl0[:] = self.zl
        # La primera condición detecta un límite móvil que ya rebasó el estado;
        # la segunda conserva el anti-windup usual cuando la derivada apunta
        # hacia fuera del intervalo.
        self.zu[:] = np.logical_or(
            np.greater(self.u.v, upper_v),
            np.logical_and(np.greater_equal(self.u.v, upper_v),
                           np.greater_equal(self.state.e, 0)),
        )
        self.zl[:] = np.logical_or(
            np.less(self.u.v, lower_v),
            np.logical_and(np.less_equal(self.u.v, lower_v),
                           np.less_equal(self.state.e, 0)),
        )
        if niter > self.niter_lock:
            self.zu[:] = np.logical_or(self.zu0, self.zu)
            self.zl[:] = np.logical_or(self.zl0, self.zl)

        self.zi[:] = np.logical_not(np.logical_or(self.zu, self.zl))
        self.x_set = list()
        if not np.all(self.zi):
            idx = np.where(self.zi == 0)
            self.state.e[:] = self.state.e * self.zi
            self.state.v[:] = (self.state.v * self.zi +
                               upper_v * self.zu + lower_v * self.zl)
            self.x_set.append((self.state.a[idx], self.state.v[idx], 0))


ANGLE_LIMIT_TOL = 1e-7  # rad; solo para delta_IT_lim
class TolerantAngleAntiWindup(AntiWindup):
    """Anti-windup que también proyecta el estado si un límite móvil lo rebasa."""

    def check_eq(self, allow_adjust=True, adjust_lower=False, adjust_upper=False,
                 is_init=False, niter=0, **kwargs):
        upper_v = -self.upper.v if self.sign_upper.v == -1 else self.upper.v
        lower_v = -self.lower.v if self.sign_lower.v == -1 else self.lower.v

        self.zu0[:] = self.zu
        self.zl0[:] = self.zl
        # La primera condición detecta un límite móvil que ya rebasó el estado;
        # la segunda conserva el anti-windup usual cuando la derivada apunta
        # hacia fuera del intervalo.
        self.zu[:] = np.logical_or(
            np.greater(self.u.v, upper_v + ANGLE_LIMIT_TOL),
            np.logical_and(np.greater_equal(self.u.v, upper_v),
                           np.greater_equal(self.state.e, 0)),
        )
        self.zl[:] = np.logical_or(
            np.less(self.u.v, lower_v - ANGLE_LIMIT_TOL),
            np.logical_and(np.less_equal(self.u.v, lower_v),
                           np.less_equal(self.state.e, 0)),
        )
        if niter > self.niter_lock:
            self.zu[:] = np.logical_or(self.zu0, self.zu)
            self.zl[:] = np.logical_or(self.zl0, self.zl)

        self.zi[:] = np.logical_not(np.logical_or(self.zu, self.zl))
        self.x_set = list()
        if not np.all(self.zi):
            idx = np.where(self.zi == 0)
            self.state.e[:] = self.state.e * self.zi
            self.state.v[:] = (self.state.v * self.zi +
                               upper_v * self.zu + lower_v * self.zl)
            self.x_set.append((self.state.a[idx], self.state.v[idx], 0))


class REGFMB1_ANData(ModelData):
    """Input data and parameters for REGFMB1_AN (per unit on inverter base Sn)."""

    def __init__(self):
        super().__init__()
        self.bus = IdxParam(model='Bus', info='interface bus id', mandatory=True)
        self.gen = IdxParam(model='StaticGen', info='static generator index', mandatory=True)

        self.Sn = NumParam(default=100.0, tex_name='S_n', unit='MVA',
                           info='inverter MVA rating', non_zero=True)
        self.fn = NumParam(default=60.0, tex_name='f_n', unit='Hz', info='rated frequency')

        # Coupling impedance on inverter base
        self.Re = NumParam(default=0.0, tex_name='R_e', unit='p.u.',
                           info='coupling resistance on inverter base')
        self.XL = NumParam(default=0.1, tex_name='X_L', unit='p.u.',
                           info='coupling reactance on inverter base', non_zero=True)

        # Operating mode flags (PSS/E ICONs)
        self.wFlag = NumParam(default=0, info='0: use wm; 1: use wPLL for P-f droop', unit='bool')
        self.VdrpFlag = NumParam(default=0, info='0: Q droop; 1: Iq droop (virtual impedance)', unit='bool')
        self.QVFlag = NumParam(default=1, info='0: Qref mode; 1: Vref mode', unit='bool')
        self.PQFlag = NumParam(default=1, info='0: Q priority; 1: P priority for steady limits', unit='bool')
        self.FFlag = NumParam(default=1, info='0: P-f droop disabled; 1: P-f droop enabled', unit='bool')
        self.ESFlag = NumParam(default=1, info='0: non-battery; 1: battery storage (allows neg Id limit)', unit='bool')

        # VSM Swing and Damping Parameters
        self.H = NumParam(default=0.5, tex_name='H', unit='s',
                          info='virtual inertia time constant (s)', non_zero=True)
        self.D1 = NumParam(default=0.0, tex_name='D_1', unit='p.u.', info='steady-state damping')
        self.D2 = NumParam(default=100.0, tex_name='D_2', unit='p.u.', info='transient damping gain')
        self.wD = NumParam(default=50.0, tex_name=r'\omega_D', unit='rad/s',
                           info='washout filter cutoff frequency', non_zero=True)
        self.Dwmax = NumParam(default=0.05, tex_name=r'\Delta\omega_{max}', unit='p.u.',
                              info='upper limit of Dwm')
        self.Dwmin = NumParam(default=-0.05, tex_name=r'\Delta\omega_{min}', unit='p.u.',
                              info='lower limit of Dwm')

        # Power-Frequency Droop
        self.mp = NumParam(default=0.02, tex_name='m_p', unit='p.u.', info='P-f droop gain (pu/pu)')
        self.Tp = NumParam(default=0.02, tex_name='T_p', unit='s', info='P-f droop filter time constant')

        # PLL Parameters
        self.kpPLL = NumParam(default=0.265, tex_name='k_{pPLL}', info='PLL proportional gain')
        self.kiPLL = NumParam(default=2.65, tex_name='k_{iPLL}', unit='p.u./s', info='PLL integral gain')
        self.DwPLLmax = NumParam(default=0.2, tex_name=r'\Delta\omega_{PLLmax}', unit='p.u.',
                                 info='upper limit of PLL output')
        self.DwPLLmin = NumParam(default=-0.2, tex_name=r'\Delta\omega_{PLLmin}', unit='p.u.',
                                 info='lower limit of PLL output')
        self.VPLLfrz = NumParam(default=0.05, tex_name='V_{PLLfrz}', unit='p.u.',
                                info='voltage threshold below which PLL is frozen')

        # Voltage Control & Reactive Droop
        self.mq = NumParam(default=0.05, tex_name='m_q', unit='p.u.',
                           info='Q-V droop gain or virtual impedance')
        self.kpv = NumParam(default=0.0, tex_name='k_{pv}', info='voltage controller proportional gain')
        self.kiv = NumParam(default=5.86, tex_name='k_{iv}', unit='p.u./s',
                            info='voltage controller integral gain')

        # Current Limits
        self.ImaxSS = NumParam(default=1.2, tex_name='I_{maxSS}', unit='p.u.',
                               info='steady-state current limit on inverter base', non_zero=True)
        self.kf = NumParam(default=0.9, tex_name='k_f',
                           info='factor to determine priority limit (Iqmax or Idmax)')
        self.kI = NumParam(default=2.0, tex_name='k_I', unit='p.u./s',
                           info='integral gain for active current limiting loop')
        self.Ke = NumParam(default=1.0, tex_name='K_e',
                           info='scaling factor on Idmax for negative active current limit')
        self.ImaxF = NumParam(default=1.5, tex_name='I_{maxF}', unit='p.u.',
                              info='transient circular current limit on inverter base', non_zero=True)

        # Transducer Measurement Filter Time Constants
        self.TPf = NumParam(default=0.02, tex_name='T_{Pf}', unit='s', info='P measurement filter')
        self.TQf = NumParam(default=0.02, tex_name='T_{Qf}', unit='s', info='Q measurement filter')
        self.TVf = NumParam(default=0.02, tex_name='T_{Vf}', unit='s', info='V measurement filter')
        self.TIf = NumParam(default=0.02, tex_name='T_{If}', unit='s', info='current measurement filter')

        # Sharing ratios of associated static generator
        self.gammap = NumParam(default=1.0, tex_name=r'\gamma_P', info='P share of static generator')
        self.gammaq = NumParam(default=1.0, tex_name=r'\gamma_Q', info='Q share of static generator')


class REGFMB1_ANModel(Model):
    """Mathematical DAE formulation for REGFMB1_AN Virtual Synchronous Machine."""

    _setpoints = {'pref': 'Pref', 'qref': 'Qref', 'vref': 'Vref'}

    def __init__(self, system, config):
        super().__init__(system, config)
        self.flags.tds = True
        self.group = 'RenGen'

        # --- Helper Constants and Services ---
        self.zero = ConstService(v_str='0', tex_name='0')
        self.one = ConstService(v_str='1', tex_name='1')
        self.w0 = ConstService(v_str='2*pi*fn', tex_name=r'\omega_0')
        self.SnSb = ConstService(v_str='Sn/sys_mva', tex_name='S_n/S_b')
        self.SbSn = ConstService(v_str='sys_mva/Sn', tex_name='S_b/S_n')
        self.Z2 = ConstService(v_str='Re**2 + XL**2', tex_name='Z_L^2')

        # Read static generator power flow solution
        self.p0s = ExtService(model='StaticGen', src='p', indexer=self.gen, tex_name='P_{0s}')
        self.q0s = ExtService(model='StaticGen', src='q', indexer=self.gen, tex_name='Q_{0s}')

        # Interface to AC network bus
        self.a = ExtAlgeb(model='Bus', src='a', indexer=self.bus, tex_name=r'\theta',
                          info='terminal voltage angle', e_str='-u*Pe',
                          ename='P', tex_ename='P', is_input=True)
        self.v = ExtAlgeb(model='Bus', src='v', indexer=self.bus, tex_name='V',
                          info='terminal voltage magnitude', e_str='-u*Qe',
                          ename='Q', tex_ename='Q', is_input=True)

        # --- Power Flow Initialization on Inverter Base ---
        self.P0 = ConstService(v_str='gammap*p0s*SbSn', tex_name='P_0')
        self.Q0 = ConstService(v_str='gammaq*q0s*SbSn', tex_name='Q_0')

        # Internal EMF behind Re + j XL in local terminal frame
        self.Ere0 = ConstService(v_str='v + (Re*P0 + XL*Q0)/v', tex_name='E_{re0}')
        self.Eim0 = ConstService(v_str='(XL*P0 - Re*Q0)/v', tex_name='E_{im0}')
        self.E0 = ConstService(v_str='sqrt(Ere0**2 + Eim0**2)', tex_name='E_0')
        self.delta_IT0 = ConstService(v_str='atan2(Eim0, Ere0)', tex_name=r'\delta_{IT0}')
        self.delta_VSM0 = ConstService(v_str='a + delta_IT0', tex_name=r'\delta_{VSM0}')

        # Projections in inverter dq frame at t=0
        self.vd0 = ConstService(v_str='v*cos(delta_IT0)', tex_name='V_{d0}')
        self.vq0 = ConstService(v_str='-v*sin(delta_IT0)', tex_name='V_{q0}')
        self.Id0 = ConstService(v_str='(Re*(E0 - vd0) - XL*vq0)/Z2', tex_name='I_{d0}')
        self.Iq0 = ConstService(v_str='(-XL*(E0 - vd0) - Re*vq0)/Z2', tex_name='I_{q0}')

        # Mode switches
        self.SWw = Switcher(u=self.wFlag, options=(0, 1), tex_name='SW_w', cache=True)
        self.SWVdrp = Switcher(u=self.VdrpFlag, options=(0, 1), tex_name='SW_{Vdrp}', cache=True)
        self.SWQV = Switcher(u=self.QVFlag, options=(0, 1), tex_name='SW_{QV}', cache=True)
        self.SWPQ = Switcher(u=self.PQFlag, options=(0, 1), tex_name='SW_{PQ}', cache=True)
        self.SWES = Switcher(u=self.ESFlag, options=(0, 1), tex_name='SW_{ES}', cache=True)

        # Reference setpoints initialized for bumpless transfer (PSS/E Note 19)
        self.Pref = ConstService(v_str='P0', tex_name='P_{ref}')
        self.Qref = ConstService(v_str='SWQV_s0*Q0', tex_name='Q_{ref}')
        self.Vref = ConstService(
            v_str='SWVdrp_s0*(v + SWQV_s1*mq*Q0) + SWVdrp_s1*(v + mq*Iq0)',
            tex_name='V_{ref}'
        )

        # Initial steady-state current capacities and deltamax0
        self.ImaxSS2 = ConstService(v_str='ImaxSS**2', tex_name='I_{maxSS}^2')
        self.IdPLL0 = ConstService(
            v_str='Id0*cos(delta_IT0) - Iq0*sin(delta_IT0)', tex_name='I_{d0}^{PLL}'
        )
        self.IqPLL0 = ConstService(
            v_str='Id0*sin(delta_IT0) + Iq0*cos(delta_IT0)', tex_name='I_{q0}^{PLL}'
        )
        self.Id02 = ConstService(v_str='IdPLL0**2', tex_name='I_{d0}^2')
        self.Iq02 = ConstService(v_str='IqPLL0**2', tex_name='I_{q0}^2')

        self.IdmaxSS0 = ConstService(
            v_str='SWPQ_s1*kf*ImaxSS + SWPQ_s0*sqrt(ImaxSS**2 - Iq02)',
            tex_name='I_{dmaxSS0}'
        )
        self.IqmaxSS0 = ConstService(
            v_str='SWPQ_s0*kf*ImaxSS + SWPQ_s1*sqrt(ImaxSS**2 - Id02)',
            tex_name='I_{qmaxSS0}'
        )
        self.deltamax0 = ConstService(v_str='asin(XL*ImaxSS)', tex_name=r'\delta_{max0}')

        # Initial dynamic voltage limits
        self.Emax0 = ConstService(v_str='sqrt((v + IqmaxSS0*XL)**2 + (IdPLL0*XL)**2)', tex_name='E_{max0}')
        self.Emin0 = ConstService(v_str='sqrt((v - IqmaxSS0*XL)**2 + (IdPLL0*XL)**2)', tex_name='E_{min0}')

        # REGFMB1U usa las componentes de corriente en el marco del PLL para
        # los limitadores y para la rama Iq-V. Id/Iq, en cambio, están en el
        # marco del voltaje interno del convertidor. Rotarlas por delta_IT hace
        # equivalentes ambas implementaciones.
        self.IdPLL = Algeb(
            v_str='IdPLL0', tex_name='I_d^{PLL}',
            e_str='Id*cos(delta_IT) - Iq*sin(delta_IT) - IdPLL'
        )
        self.IqPLL = Algeb(
            v_str='IqPLL0', tex_name='I_q^{PLL}',
            e_str='Id*sin(delta_IT) + Iq*cos(delta_IT) - IqPLL'
        )

        # --- Measurement Transducer Filters (1st-order lags) ---
        self.Pinv = Lag(u='Pdev', T=self.TPf, K=1)
        self.Qinv = Lag(u='Qdev', T=self.TQf, K=1)
        self.Vinv = Lag(u='v', T=self.TVf, K=1)
        self.Idinv = Lag(u='IdPLL', T=self.TIf, K=1)
        self.Iqinv = Lag(u='IqPLL', T=self.TIf, K=1)

        # =====================================================================
        # 1. PLL Block with Freeze Logic (Figure 4)
        # =====================================================================
        self.VqPLL = Algeb(v_str='0', tex_name='V_q^{PLL}',
                           e_str='-u*v*sin(delta_PLL - a) - VqPLL')

        # Freeze check: normal when V >= VPLLfrz, frozen when V < VPLLfrz
        self.frz_check = LessThan(u=self.v, bound=self.VPLLfrz, equal=True, tex_name='SW_{frz}')
        self.pll_en = Algeb(v_str='1', tex_name='e_{PLL}', e_str='frz_check_z0 - pll_en')

        # PLL PI Controller
        self.xi_PLL = State(v_str='0', tex_name=r'\xi_{PLL}',
                            info='PLL PI state', e_str='pll_en * kiPLL * VqPLL')
        self.DwPLL_raw = Algeb(v_str='0', tex_name=r'\Delta\omega_{PLL}^{raw}',
                               e_str='kpPLL*VqPLL + xi_PLL - DwPLL_raw')
        self.DwPLL = GainLimiter(u='DwPLL_raw', K=1, R=1,
                                 lower=self.DwPLLmin, upper=self.DwPLLmax)
        self.delta_PLL = State(v_str='a', tex_name=r'\delta_{PLL}', unit='rad',
                               info='PLL grid angle', e_str='pll_en * w0 * DwPLL_y')

        # =====================================================================
        # 2. Virtual Synchronous Machine (VSM) Swing & Damping (Figure 2)
        # =====================================================================
        # Droop speed selection: wm (wFlag=0) or wPLL (wFlag=1)
        self.Dw_droop = Algeb(v_str='0', tex_name=r'\Delta\omega_{droop}',
                              e_str='SWw_s0*Dwm + SWw_s1*DwPLL_y - Dw_droop')

        # Droop low-pass filter
        self.xdroop = Lag(u='Dw_droop', T=self.Tp, K=1)
        self.DP = Algeb(v_str='0', tex_name=r'\Delta P',
                        e_str='- (1.0/mp)*xdroop_y - DP')
        self.Pcmd = Algeb(v_str='P0', tex_name='P_{cmd}',
                          e_str='Pref + FFlag*DP - Pcmd')

        # Transient damping washout filter: s*D2 / (s + wD) * Dwm
        self.xD2 = State(v_str='0', tex_name='x_{D2}',
                         info='transient damping washout state', e_str='wD*(Dwm - xD2)')
        self.PD2 = Algeb(v_str='0', tex_name='P_{D2}',
                         e_str='D2*(Dwm - xD2) - PD2')

        # VSM Swing Equation (Inertia 2H)
        self.Dwm = State(v_str='0', tex_name=r'\Delta\omega_m', unit='p.u.',
                         info='VSM rotor speed deviation',
                         e_str='(Pcmd - Pinv_y - D1*Dwm - PD2) / (2.0*H)')
        self.Dwm_lim = AntiWindup(u=self.Dwm, lower=self.Dwmin, upper=self.Dwmax)
        self.Dwm_lim.allow_adjust = False

        # =====================================================================
        # 3. Active Current Limiting & Relative Angle delta_IT (Figure 6)
        # =====================================================================
        self.Iqinv2 = Algeb(v_str='Iq02', tex_name='I_{qinv}^2', e_str='Iqinv_y**2 - Iqinv2')
        self.Idinv2 = Algeb(v_str='Id02', tex_name='I_{dinv}^2', e_str='Idinv_y**2 - Idinv2')

        # Check limits for square root evaluation
        self.SW_d = LessThan(u=self.Iqinv2, bound=self.ImaxSS2, equal=True, tex_name='SW_d')
        self.SW_q = LessThan(u=self.Idinv2, bound=self.ImaxSS2, equal=True, tex_name='SW_q')

        self.IdmaxSS = Algeb(
            v_str='IdmaxSS0',
            tex_name='I_{dmaxSS}',
            e_str='SWPQ_s1*kf*ImaxSS + SWPQ_s0*sqrt(SW_d_z1*(ImaxSS2 - Iqinv2) + 1e-12) - IdmaxSS'
        )
        self.IqmaxSS = Algeb(
            v_str='IqmaxSS0',
            tex_name='I_{qmaxSS}',
            e_str='SWPQ_s0*kf*ImaxSS + SWPQ_s1*sqrt(SW_q_z1*(ImaxSS2 - Idinv2) + 1e-12) - IqmaxSS'
        )

        self.deltamax = Algeb(v_str='deltamax0', tex_name=r'\delta_{max}',
                              e_str='asin(XL*ImaxSS) - deltamax')
        self.neg_deltamax = Algeb(v_str='-deltamax0', tex_name=r'-\delta_{max}',
                                  e_str='-deltamax - neg_deltamax')

        # Limit integrators (pegged at their bounds at nominal operation)
        self.delta_ITmax = State(v_str='deltamax0', tex_name=r'\delta_{ITmax}', unit='rad',
                                 info='dynamic upper limit on delta_IT',
                                 e_str='kI * (IdmaxSS - Idinv_y)')
        self.delta_ITmax_lim = AntiWindup(u=self.delta_ITmax, lower=self.zero, upper=self.deltamax)
        self.delta_ITmax_lim.allow_adjust = False

        self.delta_ITmin = State(v_str='-SWES_s1*deltamax0', tex_name=r'\delta_{ITmin}', unit='rad',
                                 info='dynamic lower limit on delta_IT',
                                 e_str='SWES_s1 * kI * (-Ke*IdmaxSS - Idinv_y)')
        self.delta_ITmin_lim = AntiWindup(u=self.delta_ITmin, lower=self.neg_deltamax, upper=self.zero)
        self.delta_ITmin_lim.allow_adjust = False

        # VSM Relative Angle Integration: d(delta_IT)/dt = w0 * (Dwm - DwPLL)
        self.delta_IT = State(v_str='delta_IT0', tex_name=r'\delta_{IT}', unit='rad',
                              info='VSM angle relative to PLL',
                              e_str='w0 * (Dwm - DwPLL_y)')
        self.delta_IT_lim = TolerantAngleAntiWindup(u=self.delta_IT, lower=self.delta_ITmin, upper=self.delta_ITmax)
        self.delta_IT_lim.allow_adjust = False

        # Total internal angle
        self.delta_VSM = Algeb(v_str='delta_VSM0', tex_name=r'\delta_{VSM}', unit='rad',
                               e_str='delta_IT + delta_PLL - delta_VSM')

        # =====================================================================
        # 4. Q-V Droop and Voltage Controller with Dynamic Limits (Figure 3)
        # =====================================================================
        self.Vcmd = Algeb(
            v_str='v', tex_name='V_{cmd}',
            e_str='SWVdrp_s0*(Vref + mq*(Qref - Qinv_y)) + SWVdrp_s1*(Vref - mq*Iqinv_y) - Vcmd'
        )

        # Dynamic internal voltage limits from steady-state capacity (eqs 10-11)
        self.Emax = Algeb(
            v_str='Emax0', tex_name='E_{max}',
            e_str='sqrt((Vinv_y + IqmaxSS*XL)**2 + (Idinv_y*XL)**2 + 1e-12) - Emax'
        )
        self.Emin = Algeb(
            v_str='Emin0', tex_name='E_{min}',
            e_str='sqrt((Vinv_y - IqmaxSS*XL)**2 + (Idinv_y*XL)**2 + 1e-12) - Emin'
        )

        # Voltage PI Controller on (Vcmd - Vinv)
        self.xi_V = State(v_str='E0', tex_name=r'\xi_V', unit='p.u.',
                          info='voltage controller integrator',
                          e_str='kiv * (Vcmd - Vinv_y)')
        self.xi_V_lim = MovingBoundAntiWindup(u=self.xi_V, lower=self.Emin, upper=self.Emax)
        self.xi_V_lim.allow_adjust = False

        # Emax/Emin son límites móviles. AntiWindup bloquea la derivada cuando
        # el estado intenta salir, pero no sujeta la salida si el propio límite
        # se desplaza y rebasa al estado. REGFMB1U sí limita la salida de s11.
        self.xiV_out = GainLimiter(u=self.xi_V, K=1, R=1,
                                   lower=self.Emin, upper=self.Emax)

        self.Evsm = Algeb(
            v_str='E0', tex_name='E_{VSM}',
            e_str='kpv*(Vcmd - Vinv_y) + xiV_out_y - Evsm'
        )

        # =====================================================================
        # 5. Network Interface & Circular Transient Current Limiter (Figure 7)
        # =====================================================================
        # Terminal voltage in internal dq-frame (oriented to delta_VSM)
        self.vd = Algeb(v_str='vd0', tex_name='V_d',
                        e_str='u*v*cos(delta_VSM - a) - vd')
        self.vq = Algeb(v_str='vq0', tex_name='V_q',
                        e_str='-u*v*sin(delta_VSM - a) - vq')

        # Demanded currents through Re + j XL
        self.Idraw = Algeb(v_str='Id0', tex_name='I_d^{raw}',
                           e_str='(Re*(Evsm - vd) - XL*vq)/Z2 - Idraw')
        self.Iqraw = Algeb(v_str='Iq0', tex_name='I_q^{raw}',
                           e_str='(-XL*(Evsm - vd) - Re*vq)/Z2 - Iqraw')
        self.Iraw = Algeb(v_str='sqrt(Id0**2 + Iq0**2 + 1e-12)', tex_name='I^{raw}',
                          e_str='sqrt(Idraw**2 + Iqraw**2 + 1e-12) - Iraw')

        # Circular transient limiter (ImaxF)
        self.ILIM = LessThan(u=self.Iraw, bound=self.ImaxF, equal=True, tex_name='SW_I')
        self.Iscale = Algeb(v_str='1', tex_name='k_I',
                            e_str='ILIM_z1 + ILIM_z0*ImaxF/Iraw - Iscale')
        self.Id = Algeb(v_str='Id0', tex_name='I_d', e_str='Iscale*Idraw - Id')
        self.Iq = Algeb(v_str='Iq0', tex_name='I_q', e_str='Iscale*Iqraw - Iq')

        # Powers on inverter base
        self.Pdev = Algeb(v_str='P0', tex_name='P_{inv}', e_str='vd*Id + vq*Iq - Pdev')
        self.Qdev = Algeb(v_str='Q0', tex_name='Q_{inv}', e_str='-vd*Iq + vq*Id - Qdev')

        # Injections into system base
        self.Pe = Algeb(v_str='gammap*p0s', tex_name='P_e', e_str='SnSb*Pdev - Pe')
        self.Qe = Algeb(v_str='gammaq*q0s', tex_name='Q_e', e_str='SnSb*Qdev - Qe')

    def v_numeric(self, **kwargs):
        """Disable the corresponding StaticGen during dynamic simulation."""
        self.system.groups['StaticGen'].set(src='u', idx=self.gen.v, attr='v', value=0)


class REGFMB1_AN(REGFMB1_ANData, REGFMB1_ANModel):
    """UNIFI/WECC/PNNL REGFM_B1 Virtual Synchronous Machine GFM inverter model for ANDES."""

    def __init__(self, system, config):
        REGFMB1_ANData.__init__(self)
        REGFMB1_ANModel.__init__(self, system, config)


# Backward compatibility aliases
REGFMB1 = REGFMB1_AN
REGFMB1Data = REGFMB1_ANData
REGFMB1Model = REGFMB1_ANModel


def register_regfmb1_AN(system, quick=True):
    """Register the REGFMB1_AN model into an ANDES System instance.

    Parameters
    ----------
    system : andes.System
        The target system instance.
    quick : bool
        True to skip pretty-print generation for faster code generation.

    Returns
    -------
    REGFMB1_AN
        The instantiated model attached to system.REGFMB1_AN.
    """
    model_name = 'REGFMB1_AN'
    if model_name not in system.models:
        m = REGFMB1_AN(system=system, config=system._config_object)
        system.__dict__[model_name] = m
        system.models[model_name] = m
        m.config.check()
        system.__dict__[m.group].add_model(model_name, m)
        m.prepare(quick=quick)
    return system.models[model_name]


register_regfmb1 = register_regfmb1_AN


if __name__ == '__main__':
    import andes

    print("=== Testing REGFMB1_AN (VSM Grid-Forming Inverter) in ANDES ===")
    ss = andes.System()
    register_regfmb1_AN(ss)

    # Build 2-bus system: Bus 1 (Slack Grid) --- Line --- Bus 2 (GFM Inverter)
    b1 = ss.add('Bus', Vn=230)
    b2 = ss.add('Bus', Vn=230)
    slk = ss.add('Slack', bus=b1, Vn=230, a0=0.0, p0=0.0, q0=0.0)
    line = ss.add('Line', bus1=b1, bus2=b2, r=0.01, x=0.1, b=0.0, Sn=100)
    # Operating point within nominal inverter capacity: P=0.6, V=1.02
    pv = ss.add('PV', bus=b2, Vn=230, p0=0.6, v0=1.02, q0=0.0, Sn=100)

    # Add REGFMB1_AN VSM GFM inverter
    gfm = ss.add('REGFMB1_AN', bus=b2, gen=pv, Sn=100.0, XL=0.1, Re=0.0,
                 H=0.5, D1=0.0, D2=100.0, wD=50.0, mp=0.02, mq=0.05,
                 ImaxSS=1.2, kf=0.9, ImaxF=1.5)

    # 3-phase fault at bus 2 from t=1.0s to 1.1s
    flt = ss.add('Fault', bus=b2, tf=1.0, tc=1.1, xf=0.05)

    ss.setup()
    ss.PFlow.run()
    print(f"Power Flow converged: {ss.PFlow.converged}")

    ss.TDS.config.tf = 3.0
    ss.TDS.run()
    print(f"TDS converged: {ss.TDS.converged}")

    t = np.array(ss.dae.ts.t)
    pe = np.array(ss.dae.ts.y[:, ss.REGFMB1_AN.Pe.a[0]])
    qe = np.array(ss.dae.ts.y[:, ss.REGFMB1_AN.Qe.a[0]])
    iraw = np.array(ss.dae.ts.y[:, ss.REGFMB1_AN.Iraw.a[0]])
    iscale = np.array(ss.dae.ts.y[:, ss.REGFMB1_AN.Iscale.a[0]])
    dwm = np.array(ss.dae.ts.x[:, ss.REGFMB1_AN.Dwm.a[0]])
    delta_it = np.array(ss.dae.ts.x[:, ss.REGFMB1_AN.delta_IT.a[0]])

    print(f"t=0.0s (pre-fault):     Pe={pe[0]:.4f} pu, Qe={qe[0]:.4f} pu, Dwm={dwm[0]:.6f}, delta_IT={delta_it[0]:.4f} rad, Iscale={iscale[0]:.4f}")
    idx_flt = np.argmin(np.abs(t - 1.05))
    print(f"t=1.05s (during fault): Pe={pe[idx_flt]:.4f} pu, Qe={qe[idx_flt]:.4f} pu, Dwm={dwm[idx_flt]:.6f}, delta_IT={delta_it[idx_flt]:.4f} rad, Iscale={iscale[idx_flt]:.4f}")
    print(f"t=3.0s (post-fault):    Pe={pe[-1]:.4f} pu, Qe={qe[-1]:.4f} pu, Dwm={dwm[-1]:.6f}, delta_IT={delta_it[-1]:.4f} rad, Iscale={iscale[-1]:.4f}")
