This repo is use for calculating prop-firm pass rate, EV for each firm in here
Each of the trade history files (data/raw) before plotting inside this calculator must have all these fields below: 
entry_time,exit_time,entry_price,exit_price,quantity,pnl,highest_unrealized_profit*

*highest_unrealized_profit is for firms like traderLaunch to calculate Consistancy rules capped profit. Cause they 
automatically close your trade when it reached that capped profit.
If you don't have that data, can prefill with 0 for highest_unrealized_profit

Calculator should calculate:
- EOD trailing rule ✅
- Intraday trailing

Image results are shown in Prop_firm/Trader_launch/reports/figures_png
Text results are in Prop_firm/Trader_launch/reports/results

This repo covers:
Bulenox
Traderlaunch ✅
