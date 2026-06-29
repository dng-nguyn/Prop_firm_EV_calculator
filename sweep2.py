"""Sweep challenge/funded block size combinations."""
import sys, logging, subprocess
logging.disable(logging.CRITICAL)

results = []
for ch_bs in [1, 5, 7, 10]:
    for fu_bs in [1, 5, 7, 10]:
        code = """
import sys, logging
logging.disable(logging.CRITICAL)
sys.path.insert(0, '.')
import yaml
from Prop_firm.Trader_launch.Trader_launch_calculator import load_trades, TRADES_CSV, RESULTS_DIR, FIGURES_DIR, ensure_directories
from Prop_firm.Trader_launch.calculator_logic.Trader_launch_rules_filter_Challenge_phase import run as run_cr
from Prop_firm.Trader_launch.calculator_logic.Challenge_phase import run as run_ch
from Prop_firm.Trader_launch.calculator_logic.Trader_launch_rules_filter_Funded_phase import run as run_fr
from Prop_firm.Trader_launch.calculator_logic.Funded_phase import run as run_fu
from Prop_firm.Trader_launch.calculator_logic.Live_phase import run as run_li
from Prop_firm.Trader_launch.calculator_logic.Final_result import run as run_fi
ensure_directories()
trades = load_trades(TRADES_CSV)
run_cr(trades=trades)
ch = run_ch(trades=trades, mc_seed=42, mc_block_size=""" + str(ch_bs) + """, reports_dir=RESULTS_DIR, figures_dir=FIGURES_DIR)
run_fr(trades=trades)
fu = run_fu(trades=trades, challenge_result=ch, mc_seed=42, mc_block_size=""" + str(fu_bs) + """, reports_dir=RESULTS_DIR, figures_dir=FIGURES_DIR)
li = run_li(trades=trades, funded_result=fu, reports_dir=RESULTS_DIR, figures_dir=FIGURES_DIR)
run_fi(challenge_result=ch, funded_result=fu, live_result=li, reports_dir=RESULTS_DIR)
with open(str(RESULTS_DIR / 'Final_result_metadata.yaml')) as f:
    meta = yaml.safe_load(f)
ev = meta.get('final', {}).get('expected_value_per_pipeline_usd') or 0
ch_pr = meta.get('challenge', {}).get('mc_pass_rate_pct', 0)
fu_pr = meta.get('funded', {}).get('mc_pass_rate_pct', 0)
print('RESULT', """ + str(ch_bs) + ", " + str(fu_bs) + """, ev, ch_pr, fu_pr)
"""
        proc = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=60,
            cwd="/root/slop/Prop_firm_EV_calculator"
        )
        for line in proc.stdout.strip().split('\n'):
            if line.startswith("RESULT"):
                parts = line.split()
                results.append((int(parts[1]), int(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])))
        if proc.returncode != 0:
            print(f"ERROR ch={ch_bs} fu={fu_bs}: {proc.stderr[:200]}")

print(f"{'ChBlock':>8} {'FuBlock':>8} {'EV':>10} {'ChPass%':>8} {'FuPass%':>8}")
print("-" * 46)
for ch_bs, fu_bs, ev, ch_pr, fu_pr in sorted(results, key=lambda x: -x[2]):
    print(f"{ch_bs:>8} {fu_bs:>8} ${ev:>9.2f} {ch_pr:>7.2f}% {fu_pr:>7.2f}%")
