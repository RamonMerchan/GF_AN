"""Positive-sequence REGFM_A1_AN droop-controlled grid-forming inverter for ANDES.

Model designation: REGFMA1_AN (Grid-Forming Inverter adapted for ANDES).
Differentiated from PSS/E built-in 'regfma1' model.

Standard WECC/PNNL model for grid-forming inverter-based resources (IBRs)
represented as a voltage source behind reactance with:
  - P-f active power-frequency droop
  - Q-V reactive power-voltage droop
  - Overload mitigation controllers (Pmax, Pmin, Qmax, Qmin)
  - Voltage control mode selection (VFlag: internal E vs POI V control)
  - Reactive setpoint mode selection (QVFlag: Qref vs Vref interface)
  - Algebraic circular transient current limiter (ImaxF)
"""

import numpy as np
from andes.core import (Algeb, ConstService, ExtAlgeb, ExtService, IdxParam,
                        Lag, LessThan, Model, ModelData, NumParam, Switcher)
from andes.core.block import GainLimiter, PIAWHardLimit
from andes.core.var import State


class REGFMA1_ANData(ModelData):
    """Input data and parameters for REGFMA1_AN (per unit on inverter base Sn)."""

    def __init__(self):
        super().__init__()
        self.bus = IdxParam(model='Bus', info='interface bus id', mandatory=True)
        self.gen = IdxParam(model='StaticGen', info='static generator index', mandatory=True)

        self.Sn = NumParam(default=100.0, tex_name='S_n', unit='MVA',
                           info='inverter MVA rating', non_zero=True)
        self.XL = NumParam(default=0.15, tex_name='X_L', unit='p.u.',
                           info='coupling reactance on inverter base', non_zero=True)

        # Droop gains
        self.mp = NumParam(default=0.02, tex_name='m_p', unit='p.u.', info='P-f droop gain (pu/pu)')
        self.mq = NumParam(default=0.05, tex_name='m_q', unit='p.u.', info='Q-V droop gain (pu/pu)')

        # POI Voltage PI controller (active when VFlag=1)
        self.kpv = NumParam(default=0.0, tex_name='k_{pv}', info='POI voltage PI proportional gain')
        self.kiv = NumParam(default=5.86, tex_name='k_{iv}', unit='p.u./s',
                            info='POI voltage PI integral gain')

        # Internal voltage limits
        self.Emax = NumParam(default=1.15, tex_name='E_{max}', unit='p.u.',
                             info='maximum internal voltage')
        self.Emin = NumParam(default=0.0, tex_name='E_{min}', unit='p.u.',
                             info='minimum internal voltage')

        # Active power overload limits & PI mitigation
        self.Pmax = NumParam(default=1.0, tex_name='P_{max}', unit='p.u.',
                             info='maximum active power on inverter base')
        self.Pmin = NumParam(default=0.0, tex_name='P_{min}', unit='p.u.',
                             info='minimum active power on inverter base')
        self.kppmax = NumParam(default=0.01, tex_name='k_{ppmax}', info='P-limit P gain')
        self.kipmax = NumParam(default=0.1, tex_name='k_{ipmax}', unit='p.u./s', info='P-limit I gain')

        # Reactive power overload limits & PI mitigation
        self.Qmax = NumParam(default=0.6, tex_name='Q_{max}', unit='p.u.',
                             info='maximum reactive power on inverter base')
        self.Qmin = NumParam(default=-0.6, tex_name='Q_{min}', unit='p.u.',
                             info='minimum reactive power on inverter base')
        self.kpqmax = NumParam(default=3.0, tex_name='k_{pqmax}', info='Q-limit P gain')
        self.kiqmax = NumParam(default=20.0, tex_name='k_{iqmax}', unit='p.u./s', info='Q-limit I gain')

        # Measurement transducer filters
        self.TPf = NumParam(default=0.01, tex_name='T_{Pf}', unit='s', info='P measurement filter')
        self.TQf = NumParam(default=0.01, tex_name='T_{Qf}', unit='s', info='Q measurement filter')
        self.TVf = NumParam(default=0.01, tex_name='T_{Vf}', unit='s', info='V measurement filter')

        self.fn = NumParam(default=60.0, tex_name='f_n', unit='Hz', info='rated frequency')
        self.VFlag = NumParam(default=1, info='0: internal E control; 1: POI V control', unit='bool')
        self.QVFlag = NumParam(default=0, info='0: Qref interface; 1: Vref interface', unit='bool')

        # Transient current limiter
        self.ImaxF = NumParam(default=1.5, tex_name='I_{maxF}', unit='p.u.',
                              info='transient current limit on inverter base', non_zero=True)

        # Sharing ratios of associated static generator
        self.gammap = NumParam(default=1.0, tex_name=r'\gamma_P', info='P share of static generator')
        self.gammaq = NumParam(default=1.0, tex_name=r'\gamma_Q', info='Q share of static generator')


class REGFMA1_ANModel(Model):
    """Mathematical equations and network algebraic interface for REGFMA1_AN."""

    _setpoints = {'pref': 'Pref', 'qref': 'Qref', 'vref': 'Vref'}

    def __init__(self, system, config):
        super().__init__(system, config)
        self.flags.tds = True
        self.group = 'RenGen'

        # --- Helper Constants and Services ---
        self.zero = ConstService(v_str='0', tex_name='0')
        self.corr_lo = ConstService(v_str='-999', tex_name='c_{min}')
        self.corr_hi = ConstService(v_str='999', tex_name='c_{max}')
        self.w0 = ConstService(v_str='2*pi*fn', tex_name=r'\omega_0')
        self.SnSb = ConstService(v_str='Sn/sys_mva', tex_name='S_n/S_b')
        self.SbSn = ConstService(v_str='sys_mva/Sn', tex_name='S_b/S_n')

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

        # --- Power Flow Initialization (Inverter Base) ---
        self.P0 = ConstService(v_str='gammap*p0s*SbSn', tex_name='P_0')
        self.Q0 = ConstService(v_str='gammaq*q0s*SbSn', tex_name='Q_0')
        self.Ere0 = ConstService(v_str='v + XL*Q0/v', tex_name='E_{re0}')
        self.Eim0 = ConstService(v_str='XL*P0/v', tex_name='E_{im0}')
        self.E0 = ConstService(v_str='sqrt(Ere0**2 + Eim0**2)', tex_name='E_0')
        self.delta0 = ConstService(v_str='a + atan2(Eim0, Ere0)', tex_name=r'\delta_0')

        # Mode switches
        self.SWV = Switcher(u=self.VFlag, options=(0, 1), tex_name='SW_V', cache=True)
        self.SWQV = Switcher(u=self.QVFlag, options=(0, 1), tex_name='SW_{QV}', cache=True)

        # Reference setpoints initialized for bumpless transfer
        self.Pref = ConstService(v_str='P0', tex_name='P_{ref}')
        self.Qref = ConstService(v_str='SWQV_s0*Q0', tex_name='Q_{ref}')
        self.Vref = ConstService(v_str='SWV_s0*E0 + SWV_s1*v + mq*(Q0 - Qref)', tex_name='V_{ref}')

        # Measurement transducer filters (1st-order lags)
        self.Pf = Lag(u='Pe_dev', T=self.TPf, K=1)
        self.Qf = Lag(u='Qe_dev', T=self.TQf, K=1)
        self.Vf = Lag(u='v', T=self.TVf, K=1)

        # --- Overload Mitigation Controllers ---
        self.PmaxPI = PIAWHardLimit(u='Pmax-Pf_y', kp=self.kppmax, ki=self.kipmax, x0=0,
                                    aw_lower=self.corr_lo, aw_upper=self.zero,
                                    lower=self.corr_lo, upper=self.zero)
        self.PminPI = PIAWHardLimit(u='Pmin-Pf_y', kp=self.kppmax, ki=self.kipmax, x0=0,
                                    aw_lower=self.zero, aw_upper=self.corr_hi,
                                    lower=self.zero, upper=self.corr_hi)
        self.QmaxPI = PIAWHardLimit(u='Qmax-Qf_y', kp=self.kpqmax, ki=self.kiqmax, x0=0,
                                    aw_lower=self.corr_lo, aw_upper=self.zero,
                                    lower=self.corr_lo, upper=self.zero)
        self.QminPI = PIAWHardLimit(u='Qmin-Qf_y', kp=self.kpqmax, ki=self.kiqmax, x0=0,
                                    aw_lower=self.zero, aw_upper=self.corr_hi,
                                    lower=self.zero, upper=self.corr_hi)
        for ctrl in (self.PmaxPI, self.PminPI, self.QmaxPI, self.QminPI):
            ctrl.aw.allow_adjust = False
            ctrl.hl.allow_adjust = False

        # --- Active Power - Frequency Droop ---
        self.dw = Algeb(v_str='0', tex_name=r'\Delta\omega',
                        e_str='w0*(mp*(Pref-Pf_y) + PmaxPI_y + PminPI_y) - dw')
        self.delta = State(v_str='delta0', tex_name=r'\delta_{droop}', unit='rad',
                           info='droop internal voltage angle', e_str='dw')

        # --- Reactive Power - Voltage Droop & POI Voltage Control ---
        self.Vcmd = Algeb(v_str='SWV_s0*E0 + SWV_s1*v', tex_name='V_{cmd}',
                          e_str='Vref + mq*(Qref-Qf_y) + QmaxPI_y + QminPI_y - Vcmd')
        self.Edir = GainLimiter(u='Vcmd', K=1, R=1, lower=self.Emin, upper=self.Emax)
        self.VPI = PIAWHardLimit(u='SWV_s1*(Vcmd-Vf_y)', kp=self.kpv, ki=self.kiv, x0='E0',
                                 aw_lower=self.Emin, aw_upper=self.Emax,
                                 lower=self.Emin, upper=self.Emax)
        self.VPI.aw.allow_adjust = False
        self.VPI.hl.allow_adjust = False
        self.Edroop = Algeb(v_str='E0', tex_name='E_{droop}',
                            e_str='SWV_s0*Edir_y + SWV_s1*VPI_y - Edroop')

        # --- Terminal Voltage in Internal dq-Frame ---
        self.vd = Algeb(v_str='v*cos(delta0-a)', tex_name='V_d',
                        e_str='u*v*cos(delta-a) - vd')
        self.vq = Algeb(v_str='-v*sin(delta0-a)', tex_name='V_q',
                        e_str='-u*v*sin(delta-a) - vq')

        # Raw currents (injected into the network through XL)
        self.Idraw = Algeb(v_str='-vq/XL', tex_name='I_d^{raw}', e_str='-vq/XL - Idraw')
        self.Iqraw = Algeb(v_str='(vd-E0)/XL', tex_name='I_q^{raw}',
                           e_str='(vd-Edroop)/XL - Iqraw')
        self.Iraw = Algeb(v_str='sqrt(Idraw**2 + Iqraw**2 + 1e-12)', tex_name='I^{raw}',
                          e_str='sqrt(Idraw**2 + Iqraw**2 + 1e-12) - Iraw')

        # --- Circular Transient Current Limiter ---
        self.ILIM = LessThan(u=self.Iraw, bound=self.ImaxF, equal=True, tex_name='SW_I')
        self.Iscale = Algeb(v_str='1', tex_name='k_I',
                            e_str='ILIM_z1 + ILIM_z0*ImaxF/Iraw - Iscale')
        self.Id = Algeb(v_str='-vq/XL', tex_name='I_d', e_str='Iscale*Idraw - Id')
        self.Iq = Algeb(v_str='(vd-E0)/XL', tex_name='I_q', e_str='Iscale*Iqraw - Iq')

        # Inverter powers (on inverter base)
        self.Pe_dev = Algeb(v_str='P0', tex_name='P_{inv}', e_str='vd*Id + vq*Iq - Pe_dev')
        self.Qe_dev = Algeb(v_str='Q0', tex_name='Q_{inv}', e_str='-vd*Iq + vq*Id - Qe_dev')

        # Powers injected into system (on system base)
        self.Pe = Algeb(v_str='gammap*p0s', tex_name='P_e', e_str='SnSb*Pe_dev - Pe')
        self.Qe = Algeb(v_str='gammaq*q0s', tex_name='Q_e', e_str='SnSb*Qe_dev - Qe')

    def v_numeric(self, **kwargs):
        """Disable the corresponding StaticGens during dynamic simulation."""
        self.system.groups['StaticGen'].set(src='u', idx=self.gen.v, attr='v', value=0)


class REGFMA1_AN(REGFMA1_ANData, REGFMA1_ANModel):
    """WECC/PNNL REGFM_A1 droop-controlled grid-forming inverter model for ANDES."""

    def __init__(self, system, config):
        REGFMA1_ANData.__init__(self)
        REGFMA1_ANModel.__init__(self, system, config)


# Backward compatibility aliases
REGFMA1 = REGFMA1_AN
REGFMA1Data = REGFMA1_ANData
REGFMA1Model = REGFMA1_ANModel


def register_regfma1_AN(system, quick=True):
    """Register the REGFMA1_AN model into an ANDES System instance.

    Parameters
    ----------
    system : andes.System
        The target system instance.
    quick : bool
        True to skip pretty-print generation for faster code generation.

    Returns
    -------
    REGFMA1_AN
        The instantiated model attached to system.REGFMA1_AN.
    """
    model_name = 'REGFMA1_AN'
    if model_name not in system.models:
        m = REGFMA1_AN(system=system, config=system._config_object)
        system.__dict__[model_name] = m
        system.models[model_name] = m
        m.config.check()
        system.__dict__[m.group].add_model(model_name, m)
        m.prepare(quick=quick)
    return system.models[model_name]


register_regfma1 = register_regfma1_AN


if __name__ == '__main__':
    import andes

    print("--- Running 2-Bus Demo with REGFMA1_AN Grid-Forming Inverter ---")
    ss = andes.System()
    register_regfma1_AN(ss)

    # Build 2-bus system: Bus 1 (Slack Grid) --- Line --- Bus 2 (GFM Inverter)
    b1 = ss.add('Bus', Vn=230)
    b2 = ss.add('Bus', Vn=230)
    slk = ss.add('Slack', bus=b1, Vn=230, a0=0.0, p0=0.0, q0=0.0)
    line = ss.add('Line', bus1=b1, bus2=b2, r=0.01, x=0.1, b=0.0, Sn=100)
    pv = ss.add('PV', bus=b2, Vn=230, p0=0.6, v0=1.02, q0=0.0, Sn=100)

    # Add GFM inverter replacing PV
    gfm = ss.add('REGFMA1_AN', bus=b2, gen=pv, Sn=100.0, XL=0.15,
                 mp=0.02, mq=0.05, VFlag=1, QVFlag=0,
                 Pmax=1.0, Pmin=0.0, Qmax=1.2, Qmin=-1.2, ImaxF=1.4)

    # 3-phase fault at bus 2 from t=1.0s to 1.1s
    flt = ss.add('Fault', bus=b2, tf=1.0, tc=1.1, xf=0.05)

    ss.setup()
    ss.PFlow.run()
    print(f"Power Flow converged: {ss.PFlow.converged}")

    ss.TDS.config.tf = 3.0
    ss.TDS.run()
    print(f"TDS converged: {ss.TDS.converged}")

    t = np.array(ss.dae.ts.t)
    pe = np.array(ss.dae.ts.y[:, ss.REGFMA1_AN.Pe.a[0]])
    qe = np.array(ss.dae.ts.y[:, ss.REGFMA1_AN.Qe.a[0]])
    iraw = np.array(ss.dae.ts.y[:, ss.REGFMA1_AN.Iraw.a[0]])
    iscale = np.array(ss.dae.ts.y[:, ss.REGFMA1_AN.Iscale.a[0]])

    print(f"t=0.0s (pre-fault):     Pe={pe[0]:.4f} pu, Qe={qe[0]:.4f} pu, Iraw={iraw[0]:.4f} pu, Iscale={iscale[0]:.4f}")
    idx_flt = np.argmin(np.abs(t - 1.05))
    print(f"t=1.05s (during fault): Pe={pe[idx_flt]:.4f} pu, Qe={qe[idx_flt]:.4f} pu, Iraw={iraw[idx_flt]:.4f} pu, Iscale={iscale[idx_flt]:.4f}")
    print(f"t=3.0s (post-fault):    Pe={pe[-1]:.4f} pu, Qe={qe[-1]:.4f} pu, Iraw={iraw[-1]:.4f} pu, Iscale={iscale[-1]:.4f}")
