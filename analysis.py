"""Experiments and table generation.

    python analysis.py experiments    re-solve all experiments (requires Gurobi)
    python analysis.py tables         write LaTeX tables from data/results

Results are read from and written to data/results. Tables are written to tables/.
"""
from __future__ import annotations

import json, os, sys, time

import numpy as np

import config as C
from data import build_pilot_data, PilotData, annual_summary
# gurobipy is required only by the experiment command.
try:
    from model import (solve_planner, solve_waterfall, price_discovery,
                       _independent, _kpis, _joint, _agent_lp,
                       CARBON_PRICE, DH_PRICE, EPS_SPILL, EPS_DUMP)
    import model as M
    HAVE_SOLVER = True
except ImportError as _solver_err:
    HAVE_SOLVER = False
    M = None
    solve_planner = solve_waterfall = price_discovery = None
    _independent = _kpis = _joint = _agent_lp = None
    CARBON_PRICE, DH_PRICE, EPS_SPILL, EPS_DUMP = 0.10, 0.085, 1e-3, 1e-4

try:
    import cvxpy as cp
except ImportError:
    cp = None

_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_DIR, "data", "results")
TABLES = os.path.join(_DIR, "tables")
FIGURES = os.path.join(_DIR, "figures")
GEO = os.path.join(_DIR, "data")
os.makedirs(TABLES, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

ORDER = ["virum", "limerick", "vasteras", "graz", "turkey", "romania"]
DEEP = ["virum", "limerick", "vasteras", "graz"]
TEX_NAME = {"virum": "Virum", "limerick": "Limerick", "vasteras": "V\\\"aster\\aa s",
            "graz": "Graz--Berlin", "turkey": "Türkiye", "romania": "Romania"}
FIG_NAME = {"virum": "Virum (DK)", "limerick": "Limerick (IE)", "vasteras": "Västerås (SE)",
            "graz": "Graz–Berlin (DE-AT)", "turkey": "Türkiye", "romania": "Romania"}
VARS = ["DE0", "DE_M1", "DE_IND", "DE_M2", "DE_M3"]
VLAB = ["DE0", "+Shared", "+Diag", "+Pool", "+Grid"]

TABLES = os.path.join(_DIR, "tables")

def _load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def _write_tex(name, body):
    with open(os.path.join(TABLES, name), "w", encoding="utf-8") as f:
        f.write(body)
    print(f"  {name}")


def _annual_weights(T):
    hps = C.HOURS_PER_SEASON
    w = np.concatenate([[C.SEASON_WEIGHT[s] / (hps / 24.0)] * hps for s in C.SEASONS]) * C.DT
    return w[:T]


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [float(x) for x in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return o


def _slim(r):
    return {k: v for k, v in r.items() if k not in ("series",)}


def _save(name, obj):
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(_clean(obj), f, indent=1)
    print(f"  saved {name}")


def scale_storage(d, factor):
    for a in d.agents.values():
        a["batt_kwh"] *= factor; a["batt_kw"] *= factor
        a["tes_kwh"] *= factor; a["tes_kw"] *= factor
    return d


def run_main():
    print("[main] full matrix over all configurations")
    out = {}
    for pid in C.PILOTS:
        t0 = time.time()
        d = build_pilot_data(pid)
        wf, pr = solve_waterfall(d)
        sp = solve_planner(d, relax=False)
        splp = solve_planner(d, relax=True)
        pi3, tau3 = pr["DE_M3"]
        _, _, _, pa_m3 = _joint(d, True, CARBON_PRICE, DH_PRICE, detail=True)
        vol3 = sum(pa_m3[a]["sell"] for a in d.agents)     # aggregate pool volume per hour [kW]
        tau_bind = tau3 > 1e-4
        tau_cond = float(np.mean(tau3[tau_bind])) if tau_bind.any() else 0.0
        w = d.days_weight * d.dt
        tau_bind_annual = float(np.sum(w[tau_bind]))
        out[pid] = dict(
            tier=C.PILOTS[pid].tier, country=C.PILOTS[pid].country, zone=C.PILOTS[pid].zone,
            pv_kwp=sum(d.agents[a]["pv_kwp"] for a in d.agents),
            n_agents=len(d.agents), hosting=d.hosting_limit,
            variants={k: wf[k] for k in wf},
            SP=_slim(sp), SP_rel=_slim(splp),
            pi_mean=float(np.mean(pi3)), tau_mean=float(np.mean(tau3)),
            tau_cond_mean=tau_cond, tau_max=float(np.max(tau3)),
            tau_bind_h=int(round(tau_bind_annual)), tau_bind_rep=int(np.sum(tau_bind)),
            integrality_kEUR=float(sp["cost_kEUR"] - splp["cost_kEUR"]),
            simul_kWh=float(splp.get("simul_kWh", 0.0)),
            series=dict(I=sp["series"]["I"], E=sp["series"]["E"], CURT=sp["series"]["CURT"],
                        PV=sp["series"]["PV"], dem=sp["series"]["dem"], pi=pi3, tau=tau3, vol=vol3),
        )
        print(f"  {pid:10s} done in {time.time()-t0:.1f}s")
    _save("main_results.json", out)
    return out


def run_pareto(pids, fracs=None):
    """Cost of raising hourly self-sufficiency: tighten an annual import cap below
    the unconstrained optimum and record (self-sufficiency, cost)."""
    fracs = fracs or [1.0, 0.97, 0.94, 0.91, 0.88, 0.85, 0.82, 0.79, 0.76, 0.73, 0.70, 0.65, 0.60, 0.50, 0.40]
    print("[pareto] self-sufficiency cost curve")
    out = {}
    for pid in pids:
        d = build_pilot_data(pid)
        base = solve_planner(d, relax=True)
        base_imp = base["imp_MWh"] * 1000
        pts = [dict(frac=1.0, cost_kEUR=base["cost_kEUR"], SSR=base["SSR"],
                    co2_t=base["co2_t"], LPI_os=base["LPI_os"])]
        for f in fracs[1:]:
            r = solve_planner(d, import_cap=base_imp * f, relax=True)
            if (r.get("ok") and 0.0 <= r["SSR"] <= 1.0
                    and r["cost_kEUR"] <= 3.0 * base["cost_kEUR"]
                    and r["SSR"] >= base["SSR"] - 1e-3):
                pts.append(dict(frac=f, cost_kEUR=r["cost_kEUR"], SSR=r["SSR"],
                                co2_t=r["co2_t"], LPI_os=r["LPI_os"]))
        out[pid] = pts
        print(f"  {pid} done ({len(pts)} feasible points)")
    _save("pareto.json", out)
    return out


def run_sensitivity(pid="virum"):
    print(f"[sensitivity] pilot={pid}")
    out = {"pid": pid}
    base_host = build_pilot_data(pid).hosting_limit

    hosting = []
    for s in [0.4, 0.6, 0.8, 1.0, 1.5, 2.0]:
        d = build_pilot_data(pid); d.hosting_limit = base_host * s
        wf, _ = solve_waterfall(d)
        sp = solve_planner(d, relax=False)
        hosting.append(dict(scale=s, DE0=_slim(wf["DE0"]), DE_M3=_slim(wf["DE_M3"]), SP=_slim(sp)))
        print(f"  hosting x{s} done")
    out["hosting"] = hosting

    carbon = []
    for cp_ in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
        d = build_pilot_data(pid)
        wf, _ = solve_waterfall(d, carbon_price=cp_)
        sp = solve_planner(d, carbon_price=cp_, relax=False)
        carbon.append(dict(carbon=cp_, DE0=_slim(wf["DE0"]), DE_M3=_slim(wf["DE_M3"]), SP=_slim(sp)))
        print(f"  carbon {cp_} done")
    out["carbon"] = carbon

    storage = []
    for s in [0.0, 0.5, 1.0, 2.0, 3.0]:
        d = scale_storage(build_pilot_data(pid), s)
        wf, _ = solve_waterfall(d)
        sp = solve_planner(d, relax=False)
        storage.append(dict(scale=s, DE0=_slim(wf["DE0"]), DE_M3=_slim(wf["DE_M3"]), SP=_slim(sp)))
        print(f"  storage x{s} done")
    out["storage"] = storage

    res = []
    p = C.PILOTS[pid]; orig_ratio = p.ped_ratio
    for r in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
        p.ped_ratio = r
        d = build_pilot_data(pid)
        wf, _ = solve_waterfall(d)
        sp = solve_planner(d, relax=False)
        res.append(dict(ratio=r, DE0=_slim(wf["DE0"]), DE_M3=_slim(wf["DE_M3"]), SP=_slim(sp)))
        print(f"  RES ratio {r} done")
    p.ped_ratio = orig_ratio
    out["res"] = res

    flex = []
    orig_alpha = C.ALPHA_SHIFT
    for a in [0.0, 0.15, 0.30, 0.45, 0.60]:
        C.ALPHA_SHIFT = a
        d = build_pilot_data(pid)
        wf, _ = solve_waterfall(d)
        sp = solve_planner(d, relax=False)
        flex.append(dict(alpha=a, DE0=_slim(wf["DE0"]), DE_M3=_slim(wf["DE_M3"]), SP=_slim(sp)))
        print(f"  flex alpha {a} done")
    C.ALPHA_SHIFT = orig_alpha
    out["flex"] = flex

    feedin = []
    for fr in [0.2, 0.4, 0.6, 0.8, 1.0]:
        d = build_pilot_data(pid)
        d.price_exp = fr * d.price_imp
        wf, _ = solve_waterfall(d)
        sp = solve_planner(d, relax=False)
        feedin.append(dict(ratio=fr, DE0=_slim(wf["DE0"]), DE_M3=_slim(wf["DE_M3"]), SP=_slim(sp)))
        print(f"  feed-in {fr} done")
    out["feedin"] = feedin

    _save(f"sensitivity_{pid}.json", out)
    return out


def run_fullyear(pid="virum"):
    """Hourly full-year (8736 h) profile check against the four-representative-week
    model. Each season is solved as a storage-neutral convex block and aggregated."""
    print(f"[fullyear] pilot={pid} (8736 h)")
    orig = C.HOURS_PER_SEASON
    main = json.load(open(os.path.join(RESULTS, "main_results.json")))
    r = main[pid]
    rep = dict(cost_kEUR=r["SP"]["cost_kEUR"], SSR=r["SP"]["SSR"], LPI_os=r["SP"]["LPI_os"],
               curt_MWh=r["SP"]["curt_MWh"], co2_t=r["SP"]["co2_t"],
               gap_kEUR=r["variants"]["DE0"]["cost_kEUR"] - r["SP"]["cost_kEUR"],
               tau_bind_h=r["tau_bind_h"])

    def _season_slice(d, i, hps):
        sl = slice(i * hps, (i + 1) * hps)
        ags = {a: {k: (v[sl].copy() if isinstance(v, np.ndarray) else v) for k, v in ag.items()}
               for a, ag in d.agents.items()}
        return PilotData(pid=d.pid, T=hps, dt=d.dt, season=d.season[sl], days_weight=d.days_weight[sl],
                         Tamb=d.Tamb[sl], price_imp=d.price_imp[sl], price_exp=d.price_exp[sl],
                         tariff=d.tariff[sl], ef=d.ef[sl], ef_dh=d.ef_dh, agents=ags,
                         hosting_limit=d.hosting_limit)

    C.HOURS_PER_SEASON = 2184
    try:
        d = build_pilot_data(pid)
        cost = co2 = curt = enet = inet = dem = pv = bind = 0.0
        for i in range(len(C.SEASONS)):
            ds = _season_slice(d, i, 2184)
            sp = solve_planner(ds, relax=True)
            cost += sp["cost_kEUR"]; co2 += sp["co2_t"]; curt += sp["curt_MWh"]
            enet += sp["exp_MWh"]; inet += sp["imp_MWh"]; dem += sp["dem_MWh"]; pv += sp["pv_MWh"]
            E = np.array(sp["series"]["E"]); w = ds.days_weight * ds.dt
            bind += float(np.sum((E >= 0.99 * ds.hosting_limit) * w))
        a0, _, _ = _independent(d, "de0", CARBON_PRICE, DH_PRICE)
        de0 = _kpis(d, a0, "DE0", d.hosting_limit)
        fy = dict(cost_kEUR=cost, SSR=1 - inet / max(dem, 1e-9),
                  LPI_os=(enet + curt) / max(pv, 1e-9), curt_MWh=curt, co2_t=co2,
                  gap_kEUR=de0["cost_kEUR"] - cost, tau_bind_h=bind)
    finally:
        C.HOURS_PER_SEASON = orig
    _save("fullyear.json", {"pid": pid, "rep_week": rep, "full_year": fy})
    return {"pid": pid, "rep_week": rep, "full_year": fy}


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------
def _uncert_stats(recs, pid):
    keys = ["cost_SP", "SSR", "gap_kEUR", "gap_pct", "curt_SP",
            "m1_recovery_kEUR", "m1_recovery_pct", "curt_reduction_pct",
            "share_info", "share_foresight", "share_pool", "share_grid",
            "ann_gen_MWh", "ped_ratio"]
    stats = {k: dict(mean=float(np.mean([r[k] for r in recs])),
                     std=float(np.std([r[k] for r in recs])),
                     p10=float(np.percentile([r[k] for r in recs], 10)),
                     p25=float(np.percentile([r[k] for r in recs], 25)),
                     p75=float(np.percentile([r[k] for r in recs], 75)),
                     p90=float(np.percentile([r[k] for r in recs], 90))) for k in keys}
    stats["rank_preserved"] = float(np.mean([r["rank_ok"] for r in recs]))
    stats["n"] = len(recs)
    return {"pid": pid, "stats": stats, "draws": recs}


def _uncert_record(wf, s, d):
    m3 = wf["DE_M3"]                                     # equals SP by the zero-integrality-gap equivalence
    c = {k: wf[k]["cost_kEUR"] for k in ["DE0", "DE_M1", "DE_IND", "DE_M2", "DE_M3"]}
    c["SP"] = c["DE_M3"]
    gap = c["DE0"] - c["SP"]
    curt_de0 = wf["DE0"]["curt_MWh"]; curt_m3 = wf["DE_M3"]["curt_MWh"]
    asum = annual_summary(d)
    ann_gen = asum["pv_MWh"]; ped = ann_gen / max(asum["demand_eq_MWh"], 1e-9)
    return dict(
        seed_idx=s, cost_SP=c["SP"], SSR=m3["SSR"], gap_kEUR=gap, gap_pct=100 * gap / c["DE0"],
        curt_SP=m3["curt_MWh"], ann_gen_MWh=ann_gen, ped_ratio=ped,
        # recorded so the re-sized and fixed-capacity arms can be compared on the
        # inputs as well as the outputs, which is what exposes a specification gap
        elec_dem_MWh=asum["elec_dem_MWh"], heat_dem_MWh=asum["heat_dem_MWh"],
        demand_eq_MWh=asum["demand_eq_MWh"],
        m1_recovery_kEUR=c["DE0"] - c["DE_M1"],
        m1_recovery_pct=100 * (c["DE0"] - c["DE_M1"]) / gap if gap > 1e-6 else 0.0,
        curt_reduction_pct=100 * (curt_de0 - curt_m3) / curt_de0 if curt_de0 > 1e-6 else 0.0,
        share_info=(c["DE0"] - c["DE_M1"]) / gap if gap > 1e-6 else 0.0,
        share_foresight=(c["DE_M1"] - c["DE_IND"]) / gap if gap > 1e-6 else 0.0,
        share_pool=(c["DE_IND"] - c["DE_M2"]) / gap if gap > 1e-6 else 0.0,
        share_grid=(c["DE_M2"] - c["DE_M3"]) / gap if gap > 1e-6 else 0.0,
        rank_ok=int(c["DE0"] >= c["DE_M1"] - 1e-6 >= c["DE_M2"] - 1e-6 >= c["DE_M3"] - 1e-6 >= c["SP"] - 1e-6),
    )


def _uncert_draw(pid, s):
    d = build_pilot_data(pid, seed=C.SEED + 101 * s)
    wf, _ = solve_waterfall(d)
    return _uncert_record(wf, s, d)


def _fix_capacity(d, base_kwp, yield_factor=1.0):
    """Hold installed capacity at the base value; ``yield_factor`` scales the annual
    energy by an interannual weather-year multiplier so annual generation varies
    through the yield while installed capacity stays fixed."""
    for a, v in d.agents.items():
        if v["pv_kwp"] > 1e-9:
            v["PVavail"] = v["PVavail"] * (base_kwp[a] / v["pv_kwp"]) * yield_factor
            v["pv_kwp"] = base_kwp[a]
    return d


def run_uncertainty(pid="virum", seeds=100):
    """Monte-Carlo over weather, demand, and forecast-error seeds. Reports the
    distribution of the gap and the ordered mechanism shares and how often the
    qualitative ranking is preserved."""
    print(f"[uncertainty] pilot={pid} ({seeds} seeds)")
    recs = [_uncert_draw(pid, s) for s in range(seeds)]
    _save("uncertainty.json", _uncert_stats(recs, pid))
    return _uncert_stats(recs, pid)["stats"]


def run_uncertainty_resumable(pid="virum", target=100, budget_s=520):
    """Resumable Monte-Carlo. Draws are accumulated in results/uncertainty_raw.json
    and saved after each draw; the final uncertainty.json is written on reaching the
    target. Call repeatedly to top up under a wall-time budget."""
    raw_path = os.path.join(RESULTS, "uncertainty_raw.json")
    recs = json.load(open(raw_path))["draws"] if os.path.exists(raw_path) else []
    done = {r["seed_idx"] for r in recs}
    t0 = time.time(); s = 0
    while len(recs) < target:
        while s in done:
            s += 1
        if s >= target:
            break
        recs.append(_uncert_draw(pid, s)); done.add(s)
        json.dump(_clean({"pid": pid, "draws": recs}), open(raw_path, "w"), indent=1)
        print(f"  draw {len(recs)}/{target} (seed {s}) at {time.time()-t0:.0f}s")
        s += 1
        if time.time() - t0 > budget_s:
            print(f"  budget reached at {len(recs)} draws; rerun to continue")
            return len(recs)
    _save("uncertainty.json", _uncert_stats(recs, pid))
    print(f"  COMPLETE at {len(recs)} draws")
    return len(recs)


# --------------------------------------------------------------------------
# Robustness of the deployable step and the no-platform baseline
# --------------------------------------------------------------------------
def run_forecast(pid="virum"):
    """Sensitivity of the information step DE0->M1 to the assumed day-ahead
    forecast skill. The private error is swept with the shared error held at base."""
    print(f"[forecast] pilot={pid}")
    orig_p, orig_s = M.ERR_PRIV, M.ERR_SHARED
    d = build_pilot_data(pid)
    sp = solve_planner(d, relax=False)["cost_kEUR"]
    rows = []
    try:
        for ep in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
            M.ERR_PRIV = ep; M.ERR_SHARED = orig_s
            wf, _ = solve_waterfall(d)
            de0 = wf["DE0"]["cost_kEUR"]; m1 = wf["DE_M1"]["cost_kEUR"]
            gap = de0 - sp
            rows.append(dict(err_priv=ep, err_shared=orig_s, de0_kEUR=de0, m1_kEUR=m1,
                             info_kEUR=de0 - m1,
                             info_pct_gap=100 * (de0 - m1) / gap if gap > 1e-6 else 0.0,
                             gap_kEUR=gap))
            print(f"  err_priv {ep} done")
    finally:
        M.ERR_PRIV, M.ERR_SHARED = orig_p, orig_s
    _save("forecast.json", {"pid": pid, "sp_kEUR": sp, "base_priv": orig_p,
                            "base_shared": orig_s, "rows": rows})
    return rows


def run_baseline(pid="virum"):
    """Robustness of the DE0 baseline to a 48 h rolling horizon and a
    mean-preserving time-of-use network tariff."""
    print(f"[baseline] pilot={pid}")
    H = build_pilot_data(pid).hosting_limit
    sp = solve_planner(build_pilot_data(pid), relax=False)["cost_kEUR"]

    def _de0(d, horizon):
        a0, _, _ = _independent(d, "de0", CARBON_PRICE, DH_PRICE, horizon=horizon)
        return _kpis(d, a0, "DE0", H)["cost_kEUR"]

    base = _de0(build_pilot_data(pid), 24)
    h48 = _de0(build_pilot_data(pid), 48)
    d_tou = build_pilot_data(pid)
    hod = np.arange(d_tou.T) % 24
    fac = np.ones(d_tou.T)
    fac[(hod >= 17) & (hod <= 21)] = 1.6
    fac[(hod >= 0) & (hod <= 5)] = 0.6
    base_mean = float(np.mean(d_tou.tariff))
    d_tou.tariff = d_tou.tariff * fac
    d_tou.tariff = d_tou.tariff * (base_mean / max(float(np.mean(d_tou.tariff)), 1e-9))
    tou = _de0(d_tou, 24)
    g0 = base - sp
    rows = [dict(case="base_24h_flat", de0_kEUR=base, gap_kEUR=g0),
            dict(case="horizon_48h", de0_kEUR=h48, gap_kEUR=h48 - sp),
            dict(case="time_of_use", de0_kEUR=tou, gap_kEUR=tou - sp)]
    for r in rows:
        r["gap_pct_of_base"] = 100 * r["gap_kEUR"] / g0 if abs(g0) > 1e-9 else 0.0
    _save("baseline.json", {"pid": pid, "sp_kEUR": sp, "rows": rows})
    return rows


# --------------------------------------------------------------------------
# Market power and individual rationality
# --------------------------------------------------------------------------
def _strategic(pid):
    d = build_pilot_data(pid); w = d.days_weight * d.dt; A = list(d.agents.keys())
    _, pi, tau, pa = _joint(d, True, CARBON_PRICE, DH_PRICE, detail=True)
    net = {a: pa[a]["exp"] - pa[a]["imp"] for a in A}
    agg_net = sum(net.values())
    bind = tau > 1e-6
    wb = w[bind]
    denom = float(np.sum(wb * np.maximum(agg_net, 0.0)[bind]))

    def _share(x):
        return float(np.sum(wb * np.maximum(x, 0.0)[bind])) / denom if denom > 1e-9 else 0.0

    shares = {a: _share(net[a]) for a in A}
    # "largest agent" = largest actual binding-hour exporter (not largest PV);
    # VI-A/B/C tie on PV but VI-C has the smallest battery and exports most.
    L = max(A, key=lambda a: shares[a])
    cong_L = float(np.sum(w * tau * net[L])) / 1000
    bill_L = pa[L]["bill"]
    return dict(pid=pid, n_agents=len(A), largest=L,
                pv_share=float(d.agents[L]["pv_kwp"] / sum(d.agents[a]["pv_kwp"] for a in A)),
                export_share_binding=shares[L],
                hhi=float(np.sum(np.square(list(shares.values())))),
                cong_payment_kEUR=cong_L, bill_kEUR=bill_L,
                dev_gain_bound_pct=100 * max(cong_L, 0.0) / bill_L if bill_L > 1e-9 else 0.0,
                bind_hours_rep=int(np.sum(bind)))


def run_marketpower(pids=None):
    """Market-power indicators per configuration: binding-hour net-export
    Herfindahl index, largest-agent export share, its congestion payment, and a
    loose upper bound on any unilateral withholding gain."""
    pids = pids or list(C.PILOTS)
    print(f"[marketpower] {pids}")
    rows = [_strategic(pid) for pid in pids]
    _save("marketpower.json", {"rows": rows})
    return rows


def run_pool_price(pids=None):
    """P2P clearing-price statistics that respect pool-dual non-uniqueness: the
    trade-volume-weighted mean price over active hours, the active-hour count, the
    trade-weighted seller premium above feed-in and buyer saving below the effective
    import price, and the all-hour arithmetic mean (the particular solver dual)."""
    pids = pids or list(C.PILOTS)
    print(f"[pool-price] {pids}")
    rows = []
    for pid in pids:
        d = build_pilot_data(pid); A = list(d.agents.keys()); w = d.days_weight * d.dt
        _, pi, tau, pa = _joint(d, True, CARBON_PRICE, DH_PRICE, detail=True)
        vol = sum(pa[a]["sell"] for a in A)
        act = vol > 1.0
        wv = w * vol
        tw = lambda x: float(np.sum(wv[act] * x[act]) / np.sum(wv[act])) if act.any() else 0.0
        # objective-based M3 grid alternatives from model (4): use-of-system tariff on
        # both directions, congestion price on net export, monetized carbon on import
        c_imp = d.price_imp + d.tariff + CARBON_PRICE * d.ef - tau
        r_exp = d.price_exp - d.tariff - tau
        rows.append(dict(pid=pid, pi_tw=tw(pi), n_active_h=float(np.sum(w[act])),
                         n_active_rep=int(np.sum(act)), seller_prem=tw(pi - r_exp),
                         buyer_save=tw(c_imp - pi), pi_all_solverdual=float(np.mean(pi))))
    _save("pool_price.json", {"rows": rows})
    return rows


def run_agent_table(pid="virum"):
    """Per-agent operating cost under DE0 and M3. An agent is individually rational
    to join only if its M3 bill (which carries the unrecycled congestion payment)
    does not exceed its DE0 outside option."""
    print(f"[agent-table] pilot={pid}")
    d = build_pilot_data(pid); A = list(d.agents.keys()); T = d.T
    w = d.days_weight * d.dt; H = d.hosting_limit
    rng = np.random.default_rng(C.SEED + 0)
    keys = ["imp", "exp", "PV", "chB", "disB", "DH"]
    ser = {a: {k: np.zeros(T) for k in keys} for a in A}
    windows = list(range(0, T, 24)); nwin = len(windows); SL = C.HOURS_PER_SEASON
    for a in A:
        ag = d.agents[a]
        ref = {"eB": 0.5 * ag["batt_kwh"], "eT": 0.5 * ag["tes_kwh"]}
        e0 = dict(ref)
        for wi, d0 in enumerate(windows):
            steps = list(range(d0, min(d0 + 24, T))); end = min(d0 + 24, T)
            sl = slice(d0, end); L = len(steps)
            if d0 % SL == 0:
                e0 = dict(ref)
            pv_t = ag["PVavail"][sl]; df_t = ag["Dfix"][sl]; ds_t = ag["Dshbase"][sl]; dh_t = ag["Dheat"][sl]
            fe = {q: np.clip(1 + rng.normal(0, M.ERR_PRIV, L), 0.2, None) for q in ["pv", "load", "heat"]}
            last_of_week = (end % SL == 0 or wi == nwin - 1)
            terminal = dict(ref) if last_of_week else None
            close = {"eT": ref["eT"]} if last_of_week else None
            plan = _agent_lp(d, a, steps, pv_t * fe["pv"], df_t * fe["load"],
                             ds_t * fe["load"], dh_t * fe["heat"], e0, False, CARBON_PRICE, DH_PRICE, terminal=terminal)
            real = _agent_lp(d, a, steps, pv_t, df_t, ds_t, dh_t, e0, False, CARBON_PRICE, DH_PRICE, fix=plan, terminal=close)
            for k in keys:
                ser[a][k][sl] += real[k]
            e0 = {"eB": real["eB_end"], "eT": real["eT_end"]}
    agg = {k: sum(ser[a][k] for a in A) for k in keys}
    net = agg["imp"] - agg["exp"]
    excess = np.maximum(np.maximum(-net, 0.0) - H, 0.0)
    curt_host = np.minimum(excess, agg["PV"])
    share = np.where(agg["exp"] > 1e-9, curt_host / np.maximum(agg["exp"], 1e-9), 0.0)
    de0 = {}
    for a in A:
        s = ser[a]
        exp_eff = np.maximum(s["exp"] - s["exp"] * share, 0.0)
        de0[a] = float(np.sum(w * (d.price_imp * s["imp"] - d.price_exp * exp_eff
                        + d.tariff * (s["imp"] + exp_eff) + C.DEG_COST * (s["chB"] + s["disB"])
                        + DH_PRICE * s["DH"] + CARBON_PRICE * (d.ef * s["imp"] + d.ef_dh * s["DH"])))) / 1000
    _, _, _, pa = _joint(d, True, CARBON_PRICE, DH_PRICE, detail=True)
    m3 = {a: pa[a]["bill"] for a in A}
    rows = [dict(agent=a, de0_kEUR=de0[a], m3_kEUR=m3[a], delta_kEUR=m3[a] - de0[a],
                 ir_ok=int(m3[a] <= de0[a] + 1e-6)) for a in A]
    tot_de0 = sum(de0.values()); tot_m3 = sum(m3.values())
    _save("agent_table.json", dict(pid=pid, rows=rows, tot_de0=tot_de0, tot_m3=tot_m3,
                                   community_save_kEUR=tot_de0 - tot_m3,
                                   n_ir=int(sum(r["ir_ok"] for r in rows)), n_agents=len(A)))
    return rows


def run_convergence(pid="virum"):
    """Diagnostic for the illustrative dual-decomposition price discovery. Not used
    for any presented result."""
    print(f"[convergence] pilot={pid}")
    d = build_pilot_data(pid)
    res = price_discovery(d, iters=250, gamma0=0.04)
    _, pr = solve_waterfall(d)
    pi_c, tau_c = pr["DE_M3"]
    _save("convergence.json", dict(
        pid=pid, hist=res["hist"],
        pi_err=float(np.sqrt(np.mean((res["pi"] - pi_c) ** 2))),
        tau_err=float(np.sqrt(np.mean((res["tau"] - tau_c) ** 2))),
        pi_c_mean=float(np.mean(pi_c)), tau_c_mean=float(np.mean(tau_c)),
        pi_d_mean=float(np.mean(res["pi"])), tau_d_mean=float(np.mean(res["tau"])),
        normalization=res["normalization"]))


# --------------------------------------------------------------------------
# Independent convex-solver audit (CVXPY / Clarabel)
#
# Reproduces the convex perfect-information stages for the primary configuration
# independently of the Gurobi implementation, and reports the regularizer
# sensitivity and the pool/grid attribution-order audit.
# --------------------------------------------------------------------------
def _cvx_bounds(ag):
    flow = 5.0 * (float(np.max(ag["Dfix"] + ag["Dshbase"])) + float(np.max(ag["PVavail"])) + float(ag["batt_kw"]) + 1.0)
    shift = C.W_SHIFT * float(np.mean(ag["Dshbase"])) if ag["flex"] else 0.0
    return flow, shift


def _cvx_vars(T):
    names = ["PV", "chB", "disB", "chT", "disT", "Phe", "Pres", "Qd", "DH", "dsh", "imp", "exp", "buy", "sell"]
    v = {n: cp.Variable(T, nonneg=True) for n in names}
    v["eB"] = cp.Variable(T + 1); v["eT"] = cp.Variable(T + 1); v["qsh"] = cp.Variable(T + 1)
    return v


def _cvx_agent_constraints(d, aid, v, pool_enabled):
    ag = d.agents[aid]; T, dt = d.T, d.dt
    flow, shift = _cvx_bounds(ag)
    cop = np.clip(C.COP0 + C.KAPPA_COP * (d.Tamb - 7.0), C.COP_MIN, C.COP_MAX)
    served_cap = 2.0 * ag["Dshbase"] + 1e-6 if ag["flex"] else ag["Dshbase"]
    cons = [
        v["PV"] <= ag["PVavail"], v["chB"] <= ag["batt_kw"], v["disB"] <= ag["batt_kw"],
        v["chT"] <= ag["tes_kw"], v["disT"] <= ag["tes_kw"],
        v["Phe"] <= (2e4 if ag["hp"] else 0.0), v["Pres"] <= 2e4, v["Qd"] <= ag["Qpvt"],
        v["DH"] <= (2e4 if ag["dh"] else 0.0), v["dsh"] <= served_cap,
        v["imp"] <= flow, v["exp"] <= flow,
        v["buy"] <= (flow if pool_enabled else 0.0), v["sell"] <= (flow if pool_enabled else 0.0),
        v["eB"] >= C.SOC_MIN * ag["batt_kwh"], v["eB"] <= max(C.SOC_MAX * ag["batt_kwh"], 1e-6),
        v["eT"] >= 0.0, v["eT"] <= max(ag["tes_kwh"], 1e-6),
        v["qsh"] >= -shift, v["qsh"] <= shift,
        v["PV"] + v["disB"] + v["imp"] + v["buy"]
        == ag["Dfix"] + v["dsh"] + v["chB"] + v["Phe"] + v["Pres"] + v["exp"] + v["sell"],
        cp.multiply(cop, v["Phe"]) + v["Pres"] + ag["Qpvt"] + v["disT"] + v["DH"] == ag["Dheat"] + v["chT"] + v["Qd"],
        v["eB"][1:] == v["eB"][:-1] + C.ETA_CH * v["chB"] * dt - v["disB"] * dt / C.ETA_DIS,
        v["eT"][1:] == (1.0 - C.KAPPA_TES * dt) * v["eT"][:-1] + C.ETA_TES * v["chT"] * dt - v["disT"] * dt / C.ETA_TES,
        v["qsh"][1:] == v["qsh"][:-1] + ag["Dshbase"] - v["dsh"],
    ]
    if ag["flex"]:
        cap = C.ALPHA_SHIFT * (ag["Dfix"] + ag["Dshbase"])
        cons += [v["dsh"] - ag["Dshbase"] <= cap, ag["Dshbase"] - v["dsh"] <= cap]
    for b in range(C.HOURS_PER_SEASON, T + 1, C.HOURS_PER_SEASON):
        cons += [v["eB"][b] == v["eB"][0], v["eT"][b] == v["eT"][0]]
    for b in range(0, T + 1, 24):
        cons.append(v["qsh"][b] == 0.0)
    return cons


def _cvx_objective(d, aid, v, regularization):
    ag = d.agents[aid]; w = d.days_weight * d.dt
    economic = (cp.multiply(d.price_imp, v["imp"]) - cp.multiply(d.price_exp, v["exp"])
                + cp.multiply(d.tariff, v["imp"] + v["exp"]) + C.DEG_COST * (v["chB"] + v["disB"])
                + DH_PRICE * v["DH"] + CARBON_PRICE * (cp.multiply(d.ef, v["imp"]) + d.ef_dh * v["DH"]))
    numerical = (EPS_SPILL * (ag["PVavail"] - v["PV"]) + EPS_DUMP * v["Qd"]
                 + regularization * (cp.square(v["imp"]) + cp.square(v["exp"]) + cp.square(v["buy"]) + cp.square(v["sell"])))
    return cp.sum(cp.multiply(w, economic + numerical))


def _cvx_solve_joint(d, pool_enabled, hosting_internalized, regularization, solver="CLARABEL"):
    A = list(d.agents); T = d.T
    var = {a: _cvx_vars(T) for a in A}
    cons = []; obj = 0.0
    for a in A:
        cons += _cvx_agent_constraints(d, a, var[a], pool_enabled)
        obj = obj + _cvx_objective(d, a, var[a], regularization)
    pool_c, host_c = [], []
    for t in range(T):
        if pool_enabled:
            c = sum(var[a]["sell"][t] - var[a]["buy"][t] for a in A) == 0.0
            cons.append(c); pool_c.append(c)
        if hosting_internalized:
            c = sum(var[a]["exp"][t] - var[a]["imp"][t] for a in A) <= d.hosting_limit
            cons.append(c); host_c.append(c)
    scale = 1000.0
    prob = cp.Problem(cp.Minimize(scale * obj), cons)
    prob.solve(solver=solver, verbose=False)
    if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"convex audit joint solve failed: {prob.status}")
    keys = ["imp", "exp", "PV", "chB", "disB", "DH", "Phe", "Pres", "dsh", "buy", "sell"]
    arrays = {k: sum(np.asarray(var[a][k].value, dtype=float) for a in A) for k in keys}
    w = d.days_weight * d.dt
    pi = (np.array([-float(c.dual_value) / (scale * max(w[t], 1e-12)) for t, c in enumerate(pool_c)])
          if pool_enabled else None)
    tau = (np.array([float(c.dual_value) / (scale * max(w[t], 1e-12)) for t, c in enumerate(host_c)])
           if hosting_internalized else np.zeros(T))
    return dict(raw=float(prob.value) / scale, arrays=arrays, pi=pi, tau=tau)


def _cvx_solve_independent(d, regularization, solver="CLARABEL"):
    T = d.T
    keys = ["imp", "exp", "PV", "chB", "disB", "DH", "Phe", "Pres", "dsh", "buy", "sell"]
    agg = {k: np.zeros(T) for k in keys}
    raw = 0.0
    for a in d.agents:
        v = _cvx_vars(T)
        prob = cp.Problem(cp.Minimize(1000.0 * _cvx_objective(d, a, v, regularization)),
                          _cvx_agent_constraints(d, a, v, pool_enabled=False))
        prob.solve(solver=solver, verbose=False)
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(f"convex audit independent solve failed for {a}: {prob.status}")
        raw += float(prob.value) / 1000.0
        for k in agg:
            agg[k] += np.asarray(v[k].value, dtype=float)
    return dict(raw=raw, arrays=agg, pi=None, tau=np.zeros(T))


def _cvx_metrics(d, sol, apply_hosting_recourse):
    a = sol["arrays"]; w = d.days_weight * d.dt
    net = a["imp"] - a["exp"]
    excess = np.maximum(np.maximum(-net, 0.0) - d.hosting_limit, 0.0) if apply_hosting_recourse else np.zeros(d.T)
    curtailed = np.minimum(excess, a["PV"])
    exp_eff = np.maximum(a["exp"] - curtailed, 0.0)
    operating = np.sum(w * (d.price_imp * a["imp"] - d.price_exp * exp_eff + d.tariff * (a["imp"] + exp_eff)
                            + C.DEG_COST * (a["chB"] + a["disB"]) + DH_PRICE * a["DH"]
                            + CARBON_PRICE * (d.ef * a["imp"] + d.ef_dh * a["DH"])))
    return {
        "economic_cost_kEUR": float(operating / 1000.0),
        "raw_pre_recourse_objective_kEUR": float(sol["raw"] / 1000.0),
        "pool_volume_MWh": float(np.sum(w * a["buy"]) / 1000.0),
        "pi_mean_EUR_per_kWh": float(np.mean(sol["pi"])) if sol["pi"] is not None else 0.0,
        "tau_mean_EUR_per_kWh": float(np.mean(sol["tau"])) if sol["tau"] is not None else 0.0,
        "tau_max_EUR_per_kWh": float(np.max(sol["tau"])) if sol["tau"] is not None else 0.0,
    }


def run_convex_audit(pid="virum"):
    if cp is None:
        raise ImportError("cvxpy is required for the independent audit (pip install cvxpy clarabel)")
    print(f"[convex-audit] pilot={pid}")
    d = build_pilot_data(pid)
    reg_rows = []
    for eps in [1e-7, 1e-6, 1e-5]:
        ind = _cvx_metrics(d, _cvx_solve_independent(d, eps), apply_hosting_recourse=True)
        m2 = _cvx_metrics(d, _cvx_solve_joint(d, True, False, eps), apply_hosting_recourse=True)
        m3 = _cvx_metrics(d, _cvx_solve_joint(d, True, True, eps), apply_hosting_recourse=False)
        reg_rows.append(dict(epsilon=eps, IND=ind, M2=m2, M3=m3,
                             pool_increment_kEUR=ind["economic_cost_kEUR"] - m2["economic_cost_kEUR"],
                             grid_increment_kEUR=m2["economic_cost_kEUR"] - m3["economic_cost_kEUR"]))
        print(f"  regularizer {eps:g} done")
    reg = 1e-6
    C0 = _cvx_metrics(d, _cvx_solve_independent(d, reg), apply_hosting_recourse=True)["economic_cost_kEUR"]
    Cp = _cvx_metrics(d, _cvx_solve_joint(d, True, False, reg), apply_hosting_recourse=True)["economic_cost_kEUR"]
    Cg = _cvx_metrics(d, _cvx_solve_joint(d, False, True, reg), apply_hosting_recourse=False)["economic_cost_kEUR"]
    Cpg = _cvx_metrics(d, _cvx_solve_joint(d, True, True, reg), apply_hosting_recourse=False)["economic_cost_kEUR"]
    order = dict(
        regularization=reg,
        pool_first={"pool_kEUR": C0 - Cp, "grid_kEUR": Cp - Cpg},
        grid_first={"grid_kEUR": C0 - Cg, "pool_kEUR": Cg - Cpg},
        shapley={"pool_kEUR": 0.5 * ((C0 - Cp) + (Cg - Cpg)), "grid_kEUR": 0.5 * ((C0 - Cg) + (Cp - Cpg))},
        interaction_kEUR=Cp + Cg - C0 - Cpg,  # joint minus sum of stand-alones (positive = complementary)
    )
    _save("convex_audit.json", dict(
        description="Independent CVXPY/Clarabel audit of the convex perfect-information stages.",
        regularization_sensitivity=reg_rows, order_audit=order))


def run_reg_deployable(pid="virum"):
    """Regularizer sensitivity of the deployable stages DE0 and M1. The costs and
    the information recovery should be stable across the coupling-flow regularizer."""
    print(f"[reg-deployable] pilot={pid}")
    d = build_pilot_data(pid)
    rows = []
    for eps in [1e-7, 1e-6, 1e-5]:
        wf, _ = solve_waterfall(d, regularization=eps)
        de0 = wf["DE0"]["cost_kEUR"]; m1 = wf["DE_M1"]["cost_kEUR"]; gap = de0 - wf["DE_M3"]["cost_kEUR"]
        rows.append(dict(epsilon=eps, de0_kEUR=de0, m1_kEUR=m1, info_kEUR=de0 - m1,
                         m1_recovery_pct=100 * (de0 - m1) / gap if gap > 1e-6 else 0.0, gap_kEUR=gap))
    _save("regularization_deployable.json", rows)
    return rows


def run_uncertainty_fixed(pid="virum", seeds=100):
    """Fixed-capacity Monte Carlo: installed PV held at the base value while the
    annual yield varies with a per-draw interannual weather-year multiplier, so
    annual generation varies through the yield rather than through capacity.
    Complements the re-sized study as a genuine fixed-district robustness test."""
    print(f"[uncertainty-fixed] pilot={pid} ({seeds} seeds)")
    base_kwp = {a: v["pv_kwp"] for a, v in build_pilot_data(pid).agents.items()}
    recs = []
    for s in range(seeds):
        yf = float(np.clip(np.random.default_rng(C.SEED + 101 * s + 7).normal(1.0, 0.05), 0.85, 1.15))
        d = _fix_capacity(build_pilot_data(pid, seed=C.SEED + 101 * s), base_kwp, yf)
        wf, _ = solve_waterfall(d)
        recs.append(_uncert_record(wf, s, d))
    _save("uncertainty_fixed.json", _uncert_stats(recs, pid))
    return _uncert_stats(recs, pid)["stats"]


def run_recycling(pid="virum"):
    """Congestion-revenue recycling and individual rationality. Returns each
    agent's M3 bill after equal per-capita and payment-proportional recycling and
    whether individual rationality is restored relative to DE0. Requires agent_table.json."""
    print(f"[recycling] pilot={pid}")
    d = build_pilot_data(pid); w = d.days_weight * d.dt; A = list(d.agents.keys())
    _, pi, tau, pa = _joint(d, True, CARBON_PRICE, DH_PRICE, detail=True)
    m3 = {a: pa[a]["bill"] for a in A}
    cong = {a: float(np.sum(w * tau * (pa[a]["exp"] - pa[a]["imp"]))) / 1000 for a in A}
    R = sum(cong.values()); n = len(A)
    de0 = {r["agent"]: r["de0_kEUR"] for r in json.load(open(os.path.join(RESULTS, "agent_table.json")))["rows"]}
    rows = []
    for a in A:
        eq = m3[a] - R / n
        prop = m3[a] - cong[a]
        rows.append(dict(agent=a, de0=de0[a], m3=m3[a], cong=cong[a], m3_eq=eq, m3_prop=prop,
                         ir_m3=int(m3[a] <= de0[a] + 1e-6), ir_eq=int(eq <= de0[a] + 1e-6),
                         ir_prop=int(prop <= de0[a] + 1e-6)))
    _save("recycling.json", dict(pid=pid, revenue_kEUR=R, n=n, rows=rows,
                                 n_ir_m3=sum(r["ir_m3"] for r in rows),
                                 n_ir_eq=sum(r["ir_eq"] for r in rows),
                                 n_ir_prop=sum(r["ir_prop"] for r in rows)))
    return rows



# Table generation.

def t_pilots(main):
    rows = []
    for pid in ORDER:
        r = main[pid]; s = annual_summary(build_pilot_data(pid))
        rows.append(f"{TEX_NAME[pid]} & {r['country']} & {r['zone']} & {r['n_agents']} & "
                    f"{r['pv_kwp']:.0f} & {s['pv_MWh']:.0f} & {s['elec_dem_MWh']:.0f} & "
                    f"{s['heat_dem_MWh']:.0f} & {r['hosting']:.0f} & {r['tier']} \\\\")
    _write_tex("tab_pilots.tex",
               "\\begin{tabular}{llccrrrrrl}\n\\toprule\n"
               "Configuration & Country & Zone & Agents & PV & RES & Electricity\\ & Heat & Hosting & Tier \\\\\n"
               " & & & & [kWp] & [MWh] & [MWh] & [MWh] & [kW] & \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def t_main(main):
    rows = []
    for pid in ORDER:
        r = main[pid]; sp = r["SP"]; d0 = r["variants"]["DE0"]
        gap = d0["cost_kEUR"] - sp["cost_kEUR"]
        rows.append(f"{TEX_NAME[pid]} & {sp['cost_kEUR']:.1f} & {sp['co2_t']:.1f} & "
                    f"{sp['SSR']:.2f} & {sp.get('match_rate', 0):.2f} & {sp['LPI_os']:.2f} & "
                    f"{sp['imp_MWh']:.0f} & {sp['curt_MWh']:.0f} & {gap:.2f} & "
                    # the gap is reported as a share of the no-platform cost, not the planner
                    f"{100*gap/d0['cost_kEUR']:.1f}\\% \\\\")
    _write_tex("tab_main.tex",
               "\\begin{tabular}{lrrcccrrrr}\n\\toprule\n"
               "District & Cost & CO$_2$ & SSR & HMR & LPI$_{os}$ & Import & Curtail & Gap & Gap \\\\\n"
               " & [k\\euro] & [t] & & & & [MWh] & [MWh] & [k\\euro] & \\% \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def t_waterfall(main):
    rows = []
    for pid in ORDER:
        r = main[pid]; v = r["variants"]; sp = r["SP"]
        c = {k: v[k]["cost_kEUR"] for k in VARS}; c["SP"] = sp["cost_kEUR"]
        gap0 = c["DE0"] - c["SP"]
        rec = 100 * (c["DE0"] - c["DE_M1"]) / gap0 if gap0 > 1e-6 else 0.0
        rows.append(f"{TEX_NAME[pid]} & {c['DE0']:.1f} & {c['DE_M1']:.1f} & {c['DE_IND']:.1f} & {c['DE_M2']:.1f} & "
                    f"{c['DE_M3']:.1f} & {c['SP']:.1f} & {gap0:.2f} & {rec:.0f}\\% \\\\")
    _write_tex("tab_waterfall.tex",
               "\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
               "District & DE0 & +Shared & +Diag & +Pool & +Grid & SP & Gap & Sh.\\ share \\\\\n"
               " & [k\\euro] & [k\\euro] & [k\\euro] & [k\\euro] & [k\\euro] & [k\\euro] & [k\\euro] & \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def t_prices(main):
    rows = []
    for pid in ORDER:
        r = main[pid]
        rows.append(f"{TEX_NAME[pid]} & {r['pi_mean']:.3f} & {r['tau_mean']:.4f} & "
                    f"{r.get('tau_cond_mean',0):.3f} & {r.get('tau_max',0):.3f} & {r['tau_bind_h']} \\\\")
    _write_tex("tab_prices.tex",
               "\\begin{tabular}{lrrrrr}\n\\toprule\n"
               "District & $\\bar\\pi$ & $\\bar\\tau$ (all h) & $\\bar\\tau\\,|\\,\\tau>0$ & $\\tau^{\\max}$ & $\\tau>0$ \\\\\n"
               " & [\\euro/kWh] & [\\euro/kWh] & [\\euro/kWh] & [\\euro/kWh] & [h/yr] \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def t_equiv(main):
    rows = []
    for pid in ORDER:
        r = main[pid]; sp = r["SP"]; splp = r.get("SP_rel"); m3 = r["variants"]["DE_M3"]
        gap = 0.0 if abs(r.get("integrality_kEUR", 0.0)) < 5e-8 else r["integrality_kEUR"]
        simul = 0.0 if abs(r.get("simul_kWh", 0.0)) < 5e-8 else r["simul_kWh"]
        rows.append(f"{TEX_NAME[pid]} & {sp['cost_kEUR']:.3f} & {splp['cost_kEUR']:.3f} & "
                    f"{m3['cost_kEUR']:.3f} & {gap:.4f} & {simul:.4f} \\\\")
    _write_tex("tab_equiv.tex",
               "\\begin{tabular}{lrrrrr}\n\\toprule\n"
               "District & SP & SP$_{\\mathrm{rel}}$ & M3 & SP$-$SP$_{\\mathrm{rel}}$ & Simult.\\ ch/dis \\\\\n"
               " & [k\\euro] & [k\\euro] & [k\\euro] & [k\\euro] & [MWh] \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def t_agent_ir():
    a = _load("agent_table.json")
    rows = [f"{r['agent']} & {r['de0_kEUR']:.2f} & {r['m3_kEUR']:.2f} & {r['delta_kEUR']:+.2f} & "
            f"{'yes' if r['ir_ok'] else 'no'} \\\\" for r in a["rows"]]
    comm = f"Community & {a['tot_de0']:.2f} & {a['tot_m3']:.2f} & {a['tot_m3']-a['tot_de0']:+.2f} & -- \\\\"
    _write_tex("tab_agent_ir.tex",
               "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrrc@{}}\n\\toprule\n"
               "Agent & $\\mathrm{DE0}$ bill & $\\mathrm{M3}$ bill & Change & Individually \\\\\n"
               " & [k\\euro] & [k\\euro] & [k\\euro] & rational \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\midrule\n" + comm
               + "\n\\bottomrule\n\\end{tabular*}\n")


def t_marketpower():
    rows_in = _load("marketpower.json")["rows"]
    rows = [f"{TEX_NAME[r['pid']]} & {r['n_agents']} & {r['hhi']:.2f} & {r['export_share_binding']:.2f} & "
            f"{r['cong_payment_kEUR']:.2f} & {r['dev_gain_bound_pct']:.1f} \\\\" for r in rows_in]
    _write_tex("tab_marketpower.tex",
               "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrrrr@{}}\n\\toprule\n"
               "District & Agents & HHI & Largest & Cong.\\ pay & Exposure \\\\\n"
               " & & & exp.\\ share & [k\\euro] & [\\% of bill] \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n")


def t_forecast():
    f = _load("forecast.json")
    rows = [f"{r['err_priv']:.2f} & {r['de0_kEUR']:.2f} & {r['m1_kEUR']:.2f} & {r['info_kEUR']:.2f} & "
            f"{r['info_pct_gap']:.1f} \\\\" for r in f["rows"]]
    _write_tex("tab_forecast.tex",
               "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}rrrrr@{}}\n\\toprule\n"
               "Private error s.d.\\ & DE0 cost & M1 cost & Shared-fcst gain & M1 recovery \\\\\n"
               " & [k\\euro] & [k\\euro] & [k\\euro] & [\\% of gap] \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n")


def t_uncertainty():
    u = _load("uncertainty.json")["stats"]
    pm = lambda k, d=2: f"{u[k]['mean']:.{d}f} $\\pm$ {u[k]['std']:.{d}f}"
    iqr = lambda k, d=2: f"[{u[k]['p25']:.{d}f}, {u[k]['p75']:.{d}f}]"
    rows = [
        ("Planner cost [k\\euro]", pm("cost_SP"), iqr("cost_SP")),
        ("Self-sufficiency SSR", pm("SSR", 3), iqr("SSR", 3)),
        ("Coordination gap [k\\euro]", pm("gap_kEUR"), iqr("gap_kEUR")),
        ("Coordination gap [\\% of DE0]", pm("gap_pct"), iqr("gap_pct")),
        ("Planner curtailment [MWh]", pm("curt_SP", 1), iqr("curt_SP", 1)),
        ("Shared-forecast recovery [\\% of gap]", pm("m1_recovery_pct", 1), iqr("m1_recovery_pct", 1)),
        ("Curtailment reduction at M3 [\\%]", pm("curt_reduction_pct", 1), iqr("curt_reduction_pct", 1)),
        ("Share: shared forecast (DE0$\\to$M1)", pm("share_info"), iqr("share_info")),
        ("Share: diagnostic residual (M1$\\to$IND)", pm("share_foresight"), iqr("share_foresight")),
        ("Share: P2P market (IND$\\to$M2)", pm("share_pool"), iqr("share_pool")),
        ("Share: grid price (M2$\\to$M3)", pm("share_grid"), iqr("share_grid")),
    ]
    _write_tex("tab_uncertainty.tex",
               "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrr@{}}\n\\toprule\n"
               f"Indicator ({int(u['n'])} draws) & Mean $\\pm$ s.d.\\ & IQR [p25, p75] \\\\\n\\midrule\n"
               + "\n".join(f"{k} & {m} & {q} \\\\" for k, m, q in rows)
               + f"\n\\midrule\nMechanism ranking preserved & {100*u['rank_preserved']:.0f}\\% of draws & -- \\\\"
               + "\n\\bottomrule\n\\end{tabular*}\n")


def t_fullyear():
    fy = _load("fullyear.json")
    a, b = fy["rep_week"], fy["full_year"]
    row = lambda lab, r: (f"{lab} & {r['cost_kEUR']:.1f} & {r['SSR']:.3f} & {r['LPI_os']:.3f} & "
                          f"{r['curt_MWh']:.0f} & {r['gap_kEUR']:.2f} & {r['tau_bind_h']:.0f} \\\\")
    err = lambda k: 100 * abs(b[k] - a[k]) / max(abs(a[k]), 1e-9)
    _write_tex("tab_fullyear.tex",
               "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrrrrr@{}}\n\\toprule\n"
               "Model & Cost & SSR & LPI$_{os}$ & Curtail & Gap & Cap-bind \\\\\n"
               " & [k\\euro] & & & [MWh] & [k\\euro] & [h/yr] \\\\\n\\midrule\n"
               + row("Four representative weeks", a) + "\n" + row("Full year (8736 h)", b) + "\n\\midrule\n"
               + f"Relative difference & {err('cost_kEUR'):.1f}\\% & {err('SSR'):.1f}\\% & {err('LPI_os'):.1f}\\% & "
               f"{err('curt_MWh'):.1f}\\% & {err('gap_kEUR'):.1f}\\% & {err('tau_bind_h'):.1f}\\% \\\\"
               + "\n\\bottomrule\n\\end{tabular*}\n")


def t_reg_sensitivity():
    audit = _load("convex_audit.json")
    rows = []
    for r in audit["regularization_sensitivity"]:
        m3 = r["M3"]
        rows.append(f"{r['epsilon']:.0e} & {m3['economic_cost_kEUR']:.3f} & {r['pool_increment_kEUR']:.3f} & "
                    f"{r['grid_increment_kEUR']:.3f} & {m3['pool_volume_MWh']:.1f} & "
                    f"{m3['pi_mean_EUR_per_kWh']:.4f} & {m3['tau_mean_EUR_per_kWh']:.4f} \\\\")
    _write_tex("tab_reg_sensitivity.tex",
               "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}rrrrrrr@{}}\n\\toprule\n"
               "$\\varepsilon^{\\mathrm r}$ & M3 cost & P2P increment & Grid increment & Pool volume & $\\bar\\pi$ & $\\bar\\tau$ \\\\\n"
               " & [k\\euro] & [k\\euro] & [k\\euro] & [MWh] & [\\euro/kWh] & [\\euro/kWh] \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n")


def t_order_audit():
    a = _load("convex_audit.json")["order_audit"]
    pf, gf, sh = a["pool_first"], a["grid_first"], a["shapley"]
    rows = [f"Pool first & {pf['pool_kEUR']:.3f} & {pf['grid_kEUR']:.3f} & -- \\\\",
            f"Grid price first & {gf['pool_kEUR']:.3f} & {gf['grid_kEUR']:.3f} & -- \\\\",
            f"Two-order average (Shapley) & {sh['pool_kEUR']:.3f} & {sh['grid_kEUR']:.3f} & {a['interaction_kEUR']:.3f} \\\\"]
    _write_tex("tab_order_audit.tex",
               "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrr@{}}\n\\toprule\n"
               "Attribution rule & P2P market & Grid price & Interaction \\\\\n"
               " & [k\\euro] & [k\\euro] & [k\\euro] \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n")


def t_provenance():
    rows = [
        ("Day-ahead price level and shape", "0.07--0.13 \\euro/kWh", "2023--24", "Public statistical", "ENTSO-E zonal day-ahead ranges"),
        ("Feed-in / export ratio", "0.45--0.55", "2023--24", "Assumed", "Retail feed-in vs.\\ import spread"),
        ("Use-of-system tariff", "0.028--0.040 \\euro/kWh", "2023--24", "Public statistical", "DSO volumetric network tariffs"),
        ("Grid emission factor", "0.03--0.42 kg/kWh", "2023", "Public statistical", "Zonal location-based factors"),
        ("PV specific yield", "950--1550 kWh/kWp", "TMY", "Public statistical", "PVGIS climate ranges"),
        ("Heat-pump COP curve", "3.2 at 7\\,$^\\circ$C, +0.10/$^\\circ$C, [1.8,5.5]", "--", "Assumed", "EN~14511 datasheet range"),
        ("Battery round-trip / SoC / C-rate", "0.90 / [0.15,0.85] / 0.25", "--", "Assumed", "Li-ion datasheet typical"),
        ("Battery degradation cost", "0.02 \\euro/kWh cycled", "--", "Assumed", "Levelized cycle-life estimate"),
        ("Thermal storage eff.\\ / loss / C-rate", "0.95 / 0.01\\,h$^{-1}$ / 0.5", "--", "Assumed", "Water-tank TES typical"),
        ("Shiftable load share / depth $\\alpha$ / window", "0.25 / 0.30 / 3 h", "--", "Assumed", "Demand-response literature range"),
        ("Forecast error s.d.\\ (private/shared)", "0.25 / 0.08", "--", "Assumed", "Day-ahead PV/load forecast skill"),
        ("Hosting limit (net export)", "130--260 kW", "--", "Assumed", "$\\approx$1/3 to 2/3 of a 400 kVA substation"),
        ("Installed capacities, floor areas", "per configuration", "--", "Calibrated", "Sized to positive-energy threshold"),
    ]
    _write_tex("tab_provenance.tex",
               "\\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}p{4.3cm}"
               ">{\\raggedright\\arraybackslash}p{3.5cm}cc X@{}}\n\\toprule\n"
               "Parameter family & Representative value & Base year & Status & Source type \\\\\n\\midrule\n"
               + "\n".join(f"{a} & {b} & {c} & {dd} & {e} \\\\" for a, b, c, dd, e in rows)
               + "\n\\bottomrule\n\\end{tabularx}\n")


def t_agents():
    rows = []
    for pid in DEEP:
        d = build_pilot_data(pid); p = C.PILOTS[pid]
        for a in p.agents:
            ag = d.agents[a.aid]
            tech = []
            if ag["pv_kwp"] > 0: tech.append(f"PV {ag['pv_kwp']:.0f}")
            if a.pvt_kwp > 0: tech.append(f"PVT {a.pvt_kwp:.0f}")
            if ag["hp"]: tech.append("HP")
            if ag["batt_kwh"] > 0: tech.append(f"batt {ag['batt_kwh']:.0f}")
            if ag["tes_kwh"] > 0: tech.append(f"TES {ag['tes_kwh']:.0f}")
            if ag["dh"]: tech.append("DH")
            rows.append(f"{TEX_NAME[pid]} & {a.aid} & {a.archetype.replace('_',' ')} & "
                        f"{C.ARCH[a.archetype].floor_m2:.0f} & {', '.join(tech)} \\\\")
    _write_tex("tab_agents.tex",
               "\\begin{tabularx}{\\textwidth}{@{}lll r X@{}}\n\\toprule\n"
               "District & Agent & Archetype & Floor [m$^2$] & Technologies (kWp, kWh) \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabularx}\n")


def t_zones():
    znice = {"DK2": "Denmark (DK2)", "SE3": "Sweden (SE3)", "IE": "Ireland (SEM)",
             "DE": "Germany", "AT": "Austria", "TR": "Türkiye", "RO": "Romania"}
    rows = [f"{znice.get(zk, zk)} & {z.price_mean:.3f} & {z.price_amp:.3f} & {z.export_ratio:.2f} & "
            f"{z.tariff_use:.3f} & {z.ef_mean:.3f} & {z.ef_amp:.3f} & {z.ef_dh:.3f} \\\\"
            for zk, z in C.ZONES.items()]
    _write_tex("tab_zones.tex",
               "\\begin{tabularx}{\\textwidth}{@{}l*{7}{>{\\centering\\arraybackslash}X}@{}}\n\\toprule\n"
               "Zone & $\\bar\\lambda^{\\mathrm{imp}}$ & amp. & $\\lambda^{\\mathrm{exp}}/\\lambda^{\\mathrm{imp}}$ & "
               "$\\tau^{\\mathrm{use}}$ & $\\overline{\\mathrm{EF}}^{\\mathrm{g}}$ & amp. & $\\mathrm{EF}^{\\mathrm{dh}}$ \\\\\n"
               " & [\\euro/kWh] & [\\euro/kWh] & & [\\euro/kWh] & [kg/kWh] & [kg/kWh] & [kg/kWh] \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabularx}\n")


def t_tech():
    y = C.PV_YIELD
    rows = [
        ("Heat-pump COP intercept / slope", f"{C.COP0:.1f} at 7\\,$^\\circ$C / {C.KAPPA_COP:.2f} per $^\\circ$C"),
        ("Battery round-trip efficiency", f"{C.ETA_CH*C.ETA_DIS:.2f}"),
        ("Battery state-of-charge window", f"[{C.SOC_MIN:.2f}, {C.SOC_MAX:.2f}]"),
        ("Battery power-to-energy ratio", f"{C.BATT_C_RATE:.2f} (C/{int(1/C.BATT_C_RATE)})"),
        ("Battery degradation cost", f"{C.DEG_COST:.3f} \\euro/kWh cycled"),
        ("Heat-pump COP bounds", f"[{C.COP_MIN:.1f}, {C.COP_MAX:.1f}]"),
        ("Thermal storage efficiency / standing loss", f"{C.ETA_TES*C.ETA_TES:.2f} / {C.KAPPA_TES:.2f} per h"),
        ("Thermal storage power-to-energy ratio", f"{C.TES_C_RATE:.2f}"),
        ("Shiftable-load flexibility fraction $\\alpha_a$", f"{C.ALPHA_SHIFT:.2f}, window {C.W_SHIFT} h"),
        ("District-heating price", "0.085 \\euro/kWh"),
        ("Carbon price (base)", "100 \\euro/t"),
        ("PV specific yield (DK/IE/SE/DE/AT/TR/RO)",
         f"{y['DK']}/{y['IE']}/{y['SE']}/{y['DE']}/{y['AT']}/{y['TR']}/{y['RO']} kWh/kWp/yr"),
    ]
    _write_tex("tab_tech.tex",
               "\\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}p{6.2cm}X@{}}\n\\toprule\n"
               "Parameter & Value \\\\\n\\midrule\n"
               + "\n".join(f"{k} & {v} \\\\" for k, v in rows) + "\n\\bottomrule\n\\end{tabularx}\n")


def t_recycling():
    d = _load("recycling.json")
    rows = [f"{r['agent']} & {r['de0']:.2f} & {r['m3']:.2f} & {r['m3_eq']:.2f} & {r['m3_prop']:.2f} & "
            f"{'yes' if r['ir_eq'] else 'no'} & {'yes' if r['ir_prop'] else 'no'} \\\\" for r in d["rows"]]
    _write_tex("tab_recycling.tex",
               "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrrrcc@{}}\n\\toprule\n"
               "Agent & $\\mathrm{DE0}$ & $\\mathrm{M3}$ & $\\mathrm{M3}$+equal & $\\mathrm{M3}$+prop.\\ & IR equal & IR prop.\\ \\\\\n"
               " & [k\\euro] & [k\\euro] & [k\\euro] & [k\\euro] & & \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n")


def t_reg_deployable():
    rows_in = _load("regularization_deployable.json")
    rows = [f"$10^{{{int(round(np.log10(r['epsilon'])))}}}$ & {r['de0_kEUR']:.3f} & {r['m1_kEUR']:.3f} & "
            f"{r['info_kEUR']:.3f} & {r['m1_recovery_pct']:.1f} \\\\" for r in rows_in]
    _write_tex("tab_reg_deployable.tex",
               "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}rrrrr@{}}\n\\toprule\n"
               "$\\varepsilon^{\\mathrm r}$ & DE0 cost & M1 cost & Shared-fcst gain & M1 recovery \\\\\n"
               " & [k\\euro] & [k\\euro] & [k\\euro] & [\\% of gap] \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n")


def t_montecarlo():
    rows = [
        ("Ambient temperature", "additive $\\mathcal N(0,1.2\\,^\\circ\\mathrm{C})$ per hour", "common to all agents"),
        ("Solar clearness", "$\\mathrm{Beta}(5,2)$ per hour", "common, clipped to $[0.2,1.0]$"),
        ("Electrical load", "multiplicative $1+\\mathcal N(0,0.05)$ per hour", "independent per agent"),
        ("Day-ahead price", "additive $\\mathcal N(0,0.01)$~\\euro/kWh", "common, floored at 0.01"),
        ("Emission factor", "additive $\\mathcal N(0,0.01)$~kg/kWh", "common, floored at 0.005"),
        ("Private forecast error", "multiplicative $1+\\mathcal N(0,0.25)$", "independent per agent, clipped $\\ge0.2$"),
        ("Shared forecast error", "multiplicative $1+\\mathcal N(0,0.08)$", "common to all agents, clipped $\\ge0.2$"),
    ]
    _write_tex("tab_montecarlo.tex",
               "\\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}p{4.0cm}"
               ">{\\raggedright\\arraybackslash}p{3.7cm}X@{}}\n\\toprule\n"
               "Quantity & Perturbation & Correlation and bounds \\\\\n\\midrule\n"
               + "\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in rows) + "\n\\bottomrule\n\\end{tabularx}\n")


def t_fixedcap():
    """Table 9, the re-sized design against the fixed-capacity design."""
    a = _load("uncertainty.json")["stats"]
    b = _load("uncertainty_fixed.json")["stats"]

    def cell(st, key, dec):
        if key not in st:
            return "--"
        return f"{st[key]['mean']:.{dec}f} $\\pm$ {st[key]['std']:.{dec}f}"

    rows = [
        ("Annual PV generation [MWh]", "ann_gen_MWh", 0),
        ("PED positivity ratio (RES/elec.-equiv.\\ demand)", "ped_ratio", 2),
        ("Coordination gap [k\\euro]", "gap_kEUR", 2),
        ("Coordination gap [\\% of DE0]", "gap_pct", 2),
        ("Shared-forecast recovery [\\% of gap]", "m1_recovery_pct", 1),
        ("Curtailment reduction at M3 [\\%]", "curt_reduction_pct", 1),
        ("Self-sufficiency SSR", "SSR", 3),
    ]
    body = "\n".join(f"{lab} & {cell(a, k, d)} & {cell(b, k, d)} \\\\"
                     for lab, k, d in rows)

    # the two tests that replaced the retired ranking row, computed per draw
    sh = ("share_info", "share_foresight", "share_pool", "share_grid")

    def mono(st_file):
        dr = _load(st_file)["draws"]
        return 100.0 * sum(all(x.get(k, 0.0) >= 0.0 for k in sh) for x in dr) / len(dr)

    def above(st_file):
        dr = _load(st_file)["draws"]
        return 100.0 * sum(x["share_grid"] > x["share_pool"] for x in dr) / len(dr)

    body += (f"\nStage costs monotone, $\\mathrm{{DE0}}$ to $\\mathrm{{SP}}$ & "
             f"{mono('uncertainty.json'):.0f}\\% & {mono('uncertainty_fixed.json'):.0f}\\% \\\\")
    body += (f"\nGrid price above pool within E2 & "
             f"{above('uncertainty.json'):.0f}\\% & {above('uncertainty_fixed.json'):.0f}\\% \\\\")

    n_a = len(_load("uncertainty.json")["draws"])
    n_b = len(_load("uncertainty_fixed.json")["draws"])
    _write_tex("tab_fixedcap.tex",
               "\\begin{tabular}{lrr}\n\\toprule\n"
               f"Indicator (mean $\\pm$ s.d.) & Re-sized ({n_a} draws) & "
               f"Fixed capacity ({n_b} draws) \\\\\n\\midrule\n"
               + body + "\n\\bottomrule\n\\end{tabular}\n")


def t_peaks(main):
    """Table B.11, district peaks against the net-export hosting cap.

    Three load columns, each defined explicitly, because the retired version of this
    table carried one unlabelled "peak demand" column whose definition could not be
    recovered from the cache:

      peak electrical load excluding heat-pump input, the fixed plus baseline shiftable
      load the district would draw with no electrified heat;
      peak served electrical demand including heat-pump and resistive input, which is
      what the electricity balance actually carries;
      peak net import from the planner run, the net draw at the coupling point.

    The cap is compared with peak photovoltaic output and with the unconstrained peak
    net export only. It is an export limit and the model carries no import limit, so a
    comparison against any of the load columns would not mean anything.
    """
    rows = []
    for pid in DEEP:
        r = main[pid]
        ser = r["series"]
        d = build_pilot_data(pid)
        # load excluding electrified heat, summed across agents at each step
        base_elec = sum(a["Dfix"] + a["Dshbase"] for a in d.agents.values())
        pv = np.array(ser["PV"])
        served = np.array(ser["dem"])          # includes heat-pump and resistive input
        imp = np.array(ser["I"])
        exp = np.array(ser["E"])
        curt = np.array(ser["CURT"])
        n = min(len(base_elec), len(served))
        uncon = float(np.max(exp + curt))      # export before the DSO recourse clips it
        cap = float(r["hosting"])
        rows.append(f"{TEX_NAME[pid]} & {np.max(pv):.0f} & "
                    f"{np.max(base_elec[:n]):.0f} & {np.max(served[:n]):.0f} & "
                    f"{np.max(imp):.0f} & {uncon:.0f} & {cap:.0f} & "
                    f"{cap/np.max(pv):.2f} & {cap/uncon:.2f} \\\\")
    _write_tex("tab_peaks.tex",
               "\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
               "Configuration & Peak PV & Peak elec.\\ load & Peak served & "
               "Peak net imp.\\ & Uncon.\\ peak exp.\\ & Cap & Cap/PV & "
               "Cap/exp.\\ \\\\\n"
               " & [kW] & excl.\\ HP [kW] & incl.\\ HP [kW] & [kW] & [kW] & [kW] & "
               "& \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def t_bestresponse():
    """Table C.20, unilateral withholding by the largest exporter."""
    d = _load("best_response.json")
    base = d["rows"][0]
    lab = {1.0: "none", 0.9: "90\\%", 0.8: "80\\%", 0.7: "70\\%"}
    rows = []
    for r in d["rows"]:
        if r["case"] == base["case"]:
            db, dp = "--", "--"
        else:
            diff = r["bill_kEUR"] - base["bill_kEUR"]
            db = f"{diff:+.3f}"
            dp = f"{100*diff/base['bill_kEUR']:+.2f}"
        rows.append(f"{lab.get(round(r['frac'], 2), r['case'])} & "
                    f"{r['cong_pay_kEUR']:.2f} & {r['exp_rev_kEUR']:.2f} & "
                    f"{r['bill_kEUR']:.3f} & {db} & {dp} & "
                    f"{r['tau_mean_bind']:.4f} & {r['community_cost_kEUR']:.3f} & "
                    f"{r['curt_MWh']:.1f} \\\\")
    _write_tex("tab_bestresponse.tex",
               "\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
               "Cap on binding- & Cong.\\ & Export & Bill & $\\Delta$ bill & "
               "$\\Delta$ bill & $\\bar\\tau\\,|\\,\\tau>0$ & Community & "
               "Curtail \\\\\n"
               "hour net export & pay [k\\euro] & rev.\\ [k\\euro] & [k\\euro] & "
               "[k\\euro] & [\\%] & [\\euro/kWh] & cost [k\\euro] & [MWh] \\\\\n"
               "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def generate_tables():
    main = _load("main_results.json")
    t_pilots(main); t_main(main); t_waterfall(main); t_prices(main); t_equiv(main)
    t_agents(); t_zones(); t_tech(); t_provenance(); t_montecarlo()
    t_peaks(main)
    for name, fn in [("agent_ir", t_agent_ir), ("marketpower", t_marketpower), ("forecast", t_forecast),
                     ("uncertainty", t_uncertainty), ("fullyear", t_fullyear),
                     ("reg_sensitivity", t_reg_sensitivity), ("order_audit", t_order_audit),
                     ("recycling", t_recycling), ("reg_deployable", t_reg_deployable),
                     ("fixedcap", t_fixedcap), ("bestresponse", t_bestresponse)]:
        try:
            fn()
        except (FileNotFoundError, KeyError) as e:
            print(f"  (tab_{name} skipped: {e})")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


EXPERIMENTS = [
    ("main", lambda: run_main()),
    ("pareto", lambda: run_pareto(C.DEEP_PILOTS)),
    ("sensitivity", lambda: run_sensitivity("virum")),
    ("uncertainty", lambda: run_uncertainty("virum", seeds=100)),
    ("uncertainty_fixed", lambda: run_uncertainty_fixed("virum", seeds=100)),
    ("fullyear", lambda: run_fullyear("virum")),
    ("forecast", lambda: run_forecast("virum")),
    ("baseline", lambda: run_baseline("virum")),
    ("marketpower", lambda: run_marketpower()),
    ("pool_price", lambda: run_pool_price()),
    ("agent_table", lambda: run_agent_table("virum")),
    ("recycling", lambda: run_recycling("virum")),
    ("reg_deployable", lambda: run_reg_deployable("virum")),
]


def run_experiments():
    t0 = time.time()
    for name, fn in EXPERIMENTS:
        print(f"[{name}]")
        fn()
    if cp is not None:
        run_convex_audit("virum")
    print(f"\nAll experiments done in {time.time() - t0:.0f}s")


def main(argv):
    if len(argv) < 2 or argv[1] not in ("experiments", "tables"):
        print(__doc__)
        return 1
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(TABLES, exist_ok=True)
    if argv[1] == "experiments":
        if not HAVE_SOLVER:
            print("The experiment command needs gurobipy and a Gurobi license.\n"
                  f"Import failed with: {_solver_err}\n\n"
                  "The cached results in data/results let you rebuild every table and "
                  "figure without a solver:\n"
                  "    python analysis.py tables\n"
                  "    python plottings.py figures")
            return 2
        run_experiments()
    else:
        generate_tables()
        print(f"\nTables written to {TABLES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
