import pandas as pd
from math import floor
import matplotlib.pyplot as plt
from collections import OrderedDict
from pandas.tseries.offsets import MonthEnd
from pathlib import Path

# Track withdrawals for printing and summaries
MONTHLY_WITHDRAWALS = []               # list of dicts: {"period": "YYYY-MM", "amount": float, "time": pd.Timestamp}
WITHDRAWALS_BY_MONTH = OrderedDict()   # "YYYY-MM" -> amount
WITHDRAWALS_BY_YEAR  = OrderedDict()   # YYYY -> amount

# Track liquidation events
LIQUIDATIONS = []                      # list of dicts: {"period": "YYYY-MM", "time": pd.Timestamp}
LIQUIDATIONS_BY_MONTH = OrderedDict()  # "YYYY-MM" -> count
LIQUIDATIONS_BY_YEAR  = OrderedDict()  # YYYY -> count

# Track daily loss limit breaches
DAILY_LOSS_BREACHES = []               # list of dicts: {"time": pd.Timestamp, "equity": float}

PIP_VALUE_PER_LOT = 1.0
FEE_PER_LOT = 0.0

# base directory for loading data and saving outputs
BASE_DIR = Path(__file__).resolve().parent


def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV data, standardise columns and sort by time."""
    df = pd.read_csv(file_path, delimiter=";")
    df.columns = ["time", "open", "high", "low", "close", "tick_volume"]
    df["time"] = pd.to_datetime(df["time"], format="%Y.%m.%d %H:%M")
    df = df.sort_values("time").reset_index(drop=True)
    return df


def backtest_volume_breakout(df: pd.DataFrame, params: dict):
    """Run the breakout strategy with volume confirmation and return trades and equity."""
    lookback = params["lookback"]
    vol_lookback = params["vol_lookback"]
    vol_mult = params["vol_mult"]
    risk = params["risk"]
    starting_balance = params["balance"]
    liquidation_level = params.get("liquidation_level", 90000.0)
    balance = starting_balance

    # Daily loss parameters
    initial_account = starting_balance
    max_daily_loss_pct = params.get("max_daily_loss_pct", 0.05)
    max_daily_loss = initial_account * max_daily_loss_pct

    df = df.copy()
    df["vol_avg"] = df["tick_volume"].rolling(window=vol_lookback).mean().shift(1)

    equity_curve_pnl = [balance]
    equity_time = [df["time"].iloc[0]]
    trade_log = []

    total_sl = total_tp = total_partial = total_skips = 0
    liquidated_count = 0

    # --- live tracking for prints (separate from plotting) ---
    threshold = params.get("balance", 100000.0)

    # PnL curve (raw, no withdrawals) for printing after each trade
    pnl_balance_live = 0.0

    # Account curve (applies each trade's PnL but does NOT withdraw until month-end)
    account_balance_live = threshold

    # Track when to print month-end withdrawal
    current_month = None

    in_trade = False
    direction = None
    entry_price = entry_time = stop_price = trail_stop = partial_target = None
    partial_hit = False
    lot_size = 0.0

    # Daily loss tracking
    # `start_of_day_equity` will hold equity at the start of each day
    # `min_equity_today` tracks the lowest equity seen during the day
    # `block_trading_today` is used to optionally disable new trades after a breach
    current_day = df.loc[lookback, "time"].date()
    start_of_day_equity = balance
    min_equity_today = start_of_day_equity
    prev_close = df.loc[lookback - 1, "close"] if lookback > 0 else df.loc[0, "close"]
    block_trading_today = False
    daily_loss_breached = False  # global flag to report if any breach ever occurred

    i = lookback
    while i < len(df) - 1:
        curr_time = df.loc[i, "time"]
        curr_day = curr_time.date()

        # Record start-of-day equity at midnight
        if curr_day != current_day:
            open_pnl_midnight = 0.0
            if in_trade:
                if direction == "long":
                    open_pnl_midnight = (prev_close - entry_price) * lot_size * PIP_VALUE_PER_LOT
                else:
                    open_pnl_midnight = (entry_price - prev_close) * lot_size * PIP_VALUE_PER_LOT
            start_of_day_equity = balance + open_pnl_midnight
            min_equity_today = start_of_day_equity  # reset daily min equity
            block_trading_today = False              # allow new trades in the new day
            current_day = curr_day

        if not in_trade:
            # If a daily loss breach happened earlier today, block new trades
            if block_trading_today:
                total_skips += 1
                prev_close = df.loc[i, "close"]
                i += 1
                continue

            current_close = df.loc[i, "close"]
            recent_high = df.loc[i - lookback:i - 1, "high"].max()
            recent_low = df.loc[i - lookback:i - 1, "low"].min()
            vol_avg = df.loc[i, "vol_avg"]
            curr_vol = df.loc[i, "tick_volume"]
            trigger_long = trigger_short = False

            if current_close > recent_high:
                if pd.notna(vol_avg) and curr_vol >= vol_mult * vol_avg:
                    trigger_long = True
                else:
                    total_skips += 1
            elif current_close < recent_low:
                if pd.notna(vol_avg) and curr_vol >= vol_mult * vol_avg:
                    trigger_short = True
                else:
                    total_skips += 1

            if trigger_long or trigger_short:
                in_trade = True
                direction = "long" if trigger_long else "short"
                entry_time = df.loc[i + 1, "time"]
                entry_price = df.loc[i + 1, "open"]

                if direction == "long":
                    stop_price = recent_low
                    if stop_price >= entry_price:
                        in_trade = False
                        total_skips += 1
                        prev_close = df.loc[i, "close"]
                        i += 1
                        continue
                    stop_distance = entry_price - stop_price
                    partial_target = entry_price + stop_distance
                else:
                    stop_price = recent_high
                    if stop_price <= entry_price:
                        in_trade = False
                        total_skips += 1
                        prev_close = df.loc[i, "close"]
                        i += 1
                        continue
                    stop_distance = stop_price - entry_price
                    partial_target = entry_price - stop_distance

                partial_hit = False
                trail_stop = None
                lot_size = risk / (stop_distance * PIP_VALUE_PER_LOT)
                print(
                    f"📥 ENTRY {direction.upper()} at {entry_time} | Price: {entry_price:.2f}, "
                    f"SL: {stop_price:.2f}, Risk/pos: ${risk:.2f}"
                )
                print(
                    f"    Breakout from range {lookback}-bar High/Low and volume {curr_vol:.0f} (> {vol_mult}× avg {vol_avg:.0f})"
                )
                print(
                    f"    Initial lot size ~ {lot_size:.2f} for RISK_PER_TRADE ${risk:.2f}"
                )
                prev_close = df.loc[i, "close"]
                i += 1
                continue

        if in_trade:
            curr_time = df.loc[i, "time"]
            high = df.loc[i, "high"]
            low = df.loc[i, "low"]

            if not partial_hit:
                if direction == "long" and high >= partial_target:
                    partial_hit = True
                    profit_half = risk * 0.5
                    balance += profit_half
                    trail_stop = entry_price
                    print(
                        f"    ✅ Partial TP hit at {curr_time} - closed 50% at +1R. +${profit_half:.2f} realized, stop moved to {trail_stop:.2f}"
                    )
                    equity_time.append(curr_time)
                    equity_curve_pnl.append(balance)
                elif direction == "short" and low <= partial_target:
                    partial_hit = True
                    profit_half = risk * 0.5
                    balance += profit_half
                    trail_stop = entry_price
                    print(
                        f"    ✅ Partial TP hit at {curr_time} - closed 50% at +1R. +${profit_half:.2f} realized, stop moved to {trail_stop:.2f}"
                    )
                    equity_time.append(curr_time)
                    equity_curve_pnl.append(balance)

            if partial_hit:
                if direction == "long":
                    profit_r = (high - entry_price) / (entry_price - stop_price)
                    if profit_r >= 1 + 1e-9:
                        locked_in_r = (
                            (trail_stop - entry_price) / (entry_price - stop_price)
                            if trail_stop is not None
                            else 0
                        )
                        if profit_r >= locked_in_r + 2:
                            new_locked = floor(profit_r) - 1
                            new_trail = entry_price + new_locked * (entry_price - stop_price)
                            if trail_stop is None or new_trail > trail_stop:
                                trail_stop = new_trail
                                print(
                                    f"    🔄 Trailing stop moved up to {trail_stop:.2f} (+{new_locked}R locked)"
                                )
                else:
                    profit_r = (entry_price - low) / (stop_price - entry_price)
                    if profit_r >= 1 + 1e-9:
                        locked_in_r = (
                            (entry_price - trail_stop) / (stop_price - entry_price)
                            if trail_stop is not None
                            else 0
                        )
                        if profit_r >= locked_in_r + 2:
                            new_locked = floor(profit_r) - 1
                            new_trail = entry_price - new_locked * (stop_price - entry_price)
                            if trail_stop is None or new_trail < trail_stop:
                                trail_stop = new_trail
                                print(
                                    f"    🔄 Trailing stop moved {'down' if direction=='short' else 'up'} to {trail_stop:.2f} (+{new_locked}R locked)"
                                )

            active_stop = trail_stop if partial_hit else stop_price
            stop_hit = False
            if direction == "long":
                if low <= active_stop:
                    stop_hit = True
            else:
                if high >= active_stop:
                    stop_hit = True

            if stop_hit:
                exit_time = curr_time
                exit_price = active_stop
                if not partial_hit:
                    outcome = "FULL_SL"
                    pnl = -risk
                    balance += pnl
                    total_sl += 1
                    print(
                        f"    ❌ Stop-loss hit at {exit_time} → -${risk:.2f} (Full SL)"
                    )
                    print(f"    📊 Account balance after trade: ${balance:.2f}")
                else:
                    if direction == "long":
                        remainder_profit = (
                            (active_stop - entry_price)
                            * (lot_size / 2)
                            * PIP_VALUE_PER_LOT
                        )
                    else:
                        remainder_profit = (
                            (entry_price - active_stop)
                            * (lot_size / 2)
                            * PIP_VALUE_PER_LOT
                        )
                    pnl = remainder_profit
                    balance += pnl
                    if pnl > 1e-9:
                        outcome = "FULL_TP"
                        total_tp += 1
                        print(
                            f"    🏁 Trailing stop hit at {exit_time}, locking in remaining profit. Trade outcome: Full TP"
                        )
                        print(f"    📊 Account balance after trade: ${balance:.2f}")
                    else:
                        outcome = "PARTIAL_SL"
                        total_partial += 1
                        print(
                            f"    🟡 Break-even stop hit at {exit_time} after partial. Trade outcome: Partial TP then SL"
                        )
                        print(f"    📊 Account balance after trade: ${balance:.2f}")

                trade_log.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": exit_time,
                        "direction": "BUY" if direction == "long" else "SELL",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "stop_price": stop_price,
                        "partial_price": partial_target,
                        "outcome": outcome,
                        "net_PnL": balance - starting_balance,
                    }
                )
                equity_time.append(exit_time)
                equity_curve_pnl.append(balance)

                if balance < liquidation_level:
                    liquidated_count += 1
                    print(
                        f"    💥 Account liquidated at {exit_time}. Resetting to {starting_balance:.2f}"
                    )
                    balance = starting_balance
                    equity_time.append(exit_time)
                    equity_curve_pnl.append(balance)


                in_trade = False
                direction = None
                entry_price = entry_time = None
                stop_price = trail_stop = partial_target = None
                partial_hit = False
                lot_size = 0.0

        # Daily loss check after updating trade state
        current_price = df.loc[i, "close"]
        open_pnl = 0.0
        if in_trade:
            if direction == "long":
                open_pnl = (current_price - entry_price) * lot_size * PIP_VALUE_PER_LOT
            else:
                open_pnl = (entry_price - current_price) * lot_size * PIP_VALUE_PER_LOT
        current_equity = balance + open_pnl

        # Track the lowest equity reached today
        if current_equity < min_equity_today:
            min_equity_today = current_equity

        # Check if the drop from the day's start exceeds the allowed max loss
        if (start_of_day_equity - min_equity_today) > max_daily_loss and not block_trading_today:
            print(
                f"    🚫 Daily loss limit breached at {curr_time}. Equity: ${current_equity:.2f}"
            )
            DAILY_LOSS_BREACHES.append({"time": curr_time, "equity": float(current_equity)})
            equity_time.append(curr_time)
            equity_curve_pnl.append(current_equity)
            block_trading_today = True   # block further trades for today
            daily_loss_breached = True   # mark that a breach occurred
            # Note: we do not break out of the loop; backtest continues

        prev_close = df.loc[i, "close"]
        i += 1

    # If a trade remains open at the end of the dataset, close it out
    # regardless of whether a daily loss breach occurred earlier.
    if in_trade:
        final_time = df["time"].iloc[-1]
        final_price = df["close"].iloc[-1]
        if not partial_hit:
            outcome = "FULL_SL"
            pnl = -risk
            balance += pnl
            total_sl += 1
        else:
            if direction == "long":
                remainder_profit = (
                    (final_price - entry_price)
                    * (lot_size / 2)
                    * PIP_VALUE_PER_LOT
                )
            else:
                remainder_profit = (
                    (entry_price - final_price)
                    * (lot_size / 2)
                    * PIP_VALUE_PER_LOT
                )
            pnl = remainder_profit
            balance += pnl
            if pnl > 1e-9:
                outcome = "FULL_TP"
                total_tp += 1
            else:
                outcome = "PARTIAL_SL"
                total_partial += 1
        trade_log.append(
            {
                "entry_time": entry_time,
                "exit_time": final_time,
                "direction": "BUY" if direction == "long" else "SELL",
                "entry_price": entry_price,
                "exit_price": final_price,
                "stop_price": stop_price,
                "partial_price": partial_target,
                "outcome": outcome,
                "net_PnL": balance - starting_balance,
            }
        )
        equity_time.append(final_time)
        equity_curve_pnl.append(balance)
        if balance < liquidation_level:
            liquidated_count += 1
            print(
                f"    💥 Account liquidated at {final_time}. Resetting to {starting_balance:.2f}"
            )
            balance = starting_balance
            equity_time.append(final_time)
            equity_curve_pnl.append(balance)

        print(
            f"    ⚠️ Trade open at end of data. Closing at {final_time} price {final_price:.2f}. Outcome: {outcome}"
        )
        print(f"    📊 Account balance after trade: ${balance:.2f}")

        # realised PnL of this last trade
        pnl_balance_live += pnl
        account_balance_live += pnl

        # 🖨 print both
        print(f"📈 PnL after trade: ${pnl_balance_live:,.2f}")
        print(f"🏦 Account balance after trade: ${account_balance_live:,.2f}")

        # --- month-end withdrawal print (when month changes) ---
        t = pd.Timestamp(final_time)
        m = t.to_period("M")
        if current_month is None:
            current_month = m

        # If we've moved into a new month, settle the PREVIOUS month
        if m != current_month:
            if account_balance_live > threshold:
                w = account_balance_live - threshold
                print(
                    f"💸 Withdrawal at end of {current_month.year}-{int(current_month.month):02d}: ${w:,.2f}"
                )
                account_balance_live = threshold
            current_month = m

    trade_df = pd.DataFrame(trade_log)
    trade_df.attrs["total_skips"] = total_skips
    trade_df.attrs["total_sl"] = total_sl
    trade_df.attrs["total_tp"] = total_tp
    trade_df.attrs["total_partial"] = total_partial
    trade_df.attrs["liquidated_count"] = liquidated_count

    trade_df.attrs["risk"] = risk
    trade_df.attrs["daily_loss_breached"] = daily_loss_breached
    equity_df = pd.DataFrame({"time": equity_time, "balance": equity_curve_pnl})
    return trade_df, equity_df


# === REPLACE your current apply_withdrawal_rule with this version ===
def apply_withdrawal_rule(
    equity_curve_pnl: pd.DataFrame,
    threshold: float = 100000.0,
    liquidation_level: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float, dict[int, float], int]:
    global MONTHLY_WITHDRAWALS, WITHDRAWALS_BY_MONTH, WITHDRAWALS_BY_YEAR
    global LIQUIDATIONS, LIQUIDATIONS_BY_MONTH, LIQUIDATIONS_BY_YEAR

    """Build two aligned curves:
    - PnL curve (no withdrawals): as-is, aligned to trade times.
    - Account curve (with withdrawals + optional liquidation floor):
      starts at `threshold`, moves by PnL *differences*, withdraws the
      excess above `threshold` at month-end, and (if provided) resets to
      `threshold` whenever the account drops below `liquidation_level`.

    Returns:
      (equity_pnl_aligned, equity_account, total_withdrawn,
       yearly_withdrawals, liquidations_account)
    """
    if equity_curve_pnl.empty:
        return (
            equity_curve_pnl.copy(),
            equity_curve_pnl.copy(),
            0.0,
            {},
            0,
        )

    eq = equity_curve_pnl.sort_values("time").reset_index(drop=True)
    start_pnl = float(eq.loc[0, "balance"])

    # aligned containers
    time_list: list[pd.Timestamp] = [eq.loc[0, "time"]]
    pnl_list: list[float] = [start_pnl]         # raw PnL (never resets)
    account_list: list[float] = [threshold]     # real account (with withdrawals)
    total_withdrawn = 0.0
    yearly_withdrawals: dict[int, float] = {}
    liquidations_account = 0

    last_pnl = start_pnl
    account_balance = threshold

    for i in range(1, len(eq)):
        t = eq.loc[i, "time"]
        pnl_now = float(eq.loc[i, "balance"])

        # change since last point in the raw PnL
        pnl_diff = pnl_now - last_pnl
        last_pnl = pnl_now

        # update account by same delta
        account_balance += pnl_diff

        # append both curves at this trade timestamp
        time_list.append(t)
        pnl_list.append(pnl_now)
        account_list.append(account_balance)

        # month boundary?
        cur_m = t.to_period("M")
        next_m = eq.loc[i + 1, "time"].to_period("M") if i + 1 < len(eq) else None
        month_ended = next_m is None or next_m != cur_m

        # (optional) account-side liquidation floor
        if liquidation_level is not None and account_balance < liquidation_level:
            liquidations_account += 1
            account_balance = threshold
            period = f"{t.year}-{t.month:02d}"
            LIQUIDATIONS.append({"period": period, "time": t})
            LIQUIDATIONS_BY_MONTH[period] = LIQUIDATIONS_BY_MONTH.get(period, 0) + 1
            LIQUIDATIONS_BY_YEAR[t.year] = LIQUIDATIONS_BY_YEAR.get(t.year, 0) + 1
            # log the reset as a step at the same timestamp
            time_list.append(t)
            pnl_list.append(pnl_now)
            account_list.append(account_balance)

        # month-end withdrawal (step down to threshold)
        if month_ended and account_balance > threshold:
            withdrawal = account_balance - threshold
            total_withdrawn += withdrawal

            # yearly and monthly rollups
            y = int(cur_m.year)
            yearly_withdrawals[y] = yearly_withdrawals.get(y, 0.0) + withdrawal
            period = f"{int(cur_m.year)}-{int(cur_m.month):02d}"
            WITHDRAWALS_BY_MONTH[period] = WITHDRAWALS_BY_MONTH.get(period, 0.0) + withdrawal
            MONTHLY_WITHDRAWALS.append({"period": period, "amount": float(withdrawal), "time": t})
            WITHDRAWALS_BY_YEAR[int(cur_m.year)] = WITHDRAWALS_BY_YEAR.get(int(cur_m.year), 0.0) + withdrawal

            account_balance = threshold
            time_list.append(t)
            pnl_list.append(pnl_now)
            account_list.append(account_balance)

    equity_pnl_aligned = pd.DataFrame({"time": time_list, "balance": pnl_list})
    equity_account     = pd.DataFrame({"time": time_list, "balance": account_list})

    return equity_pnl_aligned, equity_account, total_withdrawn, yearly_withdrawals, liquidations_account

def print_monthly_summary(year, month, balance, account_balance, start_balance, trades, liquidations):
    pnl = balance - start_balance
    wins = sum(1 for t in trades if t.get("outcome") != "FULL_SL")
    losses = len(trades) - wins
    sign = "+" if pnl >= 0 else "-"
    period = f"{year}-{month:02d}"
    withdrawal = WITHDRAWALS_BY_MONTH.get(period, 0.0)
    withdrawal_str = f" | Withdrawal: ${withdrawal:,.2f}" if withdrawal > 0 else ""
    print(
        f"{year}-{month:02d} | Balance: ${balance:,.2f} | Account: ${account_balance:,.2f} | "
        f"Trades: {len(trades)} | W:{wins} L:{losses} | P&L: {sign}${abs(pnl):,.2f} | "
        f"Liquidations: {liquidations}{withdrawal_str}"
    )



def analyze_results(
    trade_log: pd.DataFrame,
    equity_curve_pnl: pd.DataFrame,
    equity_curve_account: pd.DataFrame,
    df: pd.DataFrame,
    total_withdrawn: float,
    yearly_withdrawals: dict,   # or dict[int, float] if you prefer
) -> None:

    """Print a summary of results and plot both equity curves."""
    total_trades = len(trade_log)
    total_tp = (trade_log["outcome"] == "FULL_TP").sum()
    total_sl = (trade_log["outcome"] == "FULL_SL").sum()
    total_partial = (trade_log["outcome"] == "PARTIAL_SL").sum()
    total_skips = trade_log.attrs.get("total_skips", 0)
    liquidated_count = trade_log.attrs.get("liquidated_count", 0)

    wins = total_tp + total_partial
    win_rate = wins / total_trades * 100 if total_trades else 0.0
    net_profit = equity_curve_pnl["balance"].iloc[-1] - equity_curve_pnl["balance"].iloc[0]
    risk = trade_log.attrs.get("risk", 1)

    print("\n==== Strategy Performance Summary ====")
    print(f"Period tested: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
    print(f"Total Trades: {total_trades}")
    print(f"Full TP outcomes: {total_tp}")
    print(f"Full SL outcomes: {total_sl}")
    print(f"Partial TP then SL outcomes: {total_partial}")
    print(f"Skipped signals due to low volume: {total_skips}")
    print(f"Accounts liquidated: {liquidated_count}")

    print(f"Win rate (incl. partial wins): {win_rate:.1f}%")
    print(f"Net Profit: {net_profit:.2f} ({net_profit/risk:.1f}R)")
    print("======================================")
    print("\n==== Withdrawal Summary ====")
    print(f"Total withdrawals: ${total_withdrawn:.2f}")
    for year in sorted(yearly_withdrawals):
        amount = yearly_withdrawals[year]
        print(f"{year}: ${amount:.2f}")
    print("============================")
    # — Monthly withdrawals (by month) —
    if WITHDRAWALS_BY_MONTH:
        print("\n— Monthly withdrawals —")
        for period in sorted(WITHDRAWALS_BY_MONTH):
            print(f"{period}: ${WITHDRAWALS_BY_MONTH[period]:.2f}")

        # Consistency check
        _ysum = sum(yearly_withdrawals.values()) if yearly_withdrawals else 0.0
        _msum = sum(WITHDRAWALS_BY_MONTH.values()) if WITHDRAWALS_BY_MONTH else 0.0
        if abs(_ysum - total_withdrawn) > 1e-6 or abs(_msum - total_withdrawn) > 1e-6:
            print(f"WARNING: mismatch → total={total_withdrawn:.2f}, yearly_sum={_ysum:.2f}, monthly_sum={_msum:.2f}")

    print("\n==== Liquidation Summary ====")
    total_liq = sum(LIQUIDATIONS_BY_YEAR.values()) if LIQUIDATIONS_BY_YEAR else 0
    print(f"Total liquidations: {total_liq}")
    for year in sorted(LIQUIDATIONS_BY_YEAR):
        print(f"{year}: {LIQUIDATIONS_BY_YEAR[year]}")
    print("============================")
    if LIQUIDATIONS_BY_MONTH:
        print("\n— Monthly liquidations —")
        for period in sorted(LIQUIDATIONS_BY_MONTH):
            print(f"{period}: {LIQUIDATIONS_BY_MONTH[period]}")

    # — Monthly performance summary —
    eq_pnl = equity_curve_pnl.copy()
    eq_pnl["period"] = eq_pnl["time"].dt.to_period("M")
    eq_account = equity_curve_account.copy()
    eq_account["period"] = eq_account["time"].dt.to_period("M")
    trade_months = trade_log.copy()
    trade_months["period"] = trade_months["exit_time"].dt.to_period("M")

    start_balance = eq_pnl["balance"].iloc[0]
    for period in eq_pnl["period"].unique():
        pnl_end = eq_pnl[eq_pnl["period"] == period]["balance"].iloc[-1]
        account_end = eq_account[eq_account["period"] == period]["balance"].iloc[-1]
        monthly_trades = trade_months[trade_months["period"] == period].to_dict("records")
        liq = LIQUIDATIONS_BY_MONTH.get(f"{period.year}-{period.month:02d}", 0)
        print_monthly_summary(period.year, period.month, pnl_end, account_end, start_balance, monthly_trades, liq)
        start_balance = pnl_end

    plt.figure(figsize=(8, 4))
    plt.plot(
        equity_curve_pnl["time"],
        equity_curve_pnl["balance"],
        label="PnL (no withdrawals)",
        color="blue",
    )
    plt.plot(
        equity_curve_account["time"],
        equity_curve_account["balance"],
        label="Account (with withdrawals)",
        color="green",
    )
    plt.title("NAS100 Breakout Strategy Equity Curves")
    plt.xlabel("Time")
    plt.ylabel("Account Balance")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_data(BASE_DIR / "US100.cash_2024.csv")
    params = {
        "lookback": 20,
        "vol_lookback": 20,
        "vol_mult": 2.0,
        "risk": 2500.0,
        "balance": 100000.0,
        "liquidation_level": 90000.0,
        "max_daily_loss_pct": 0.05,

    }
    trades, equity_pnl_raw = backtest_volume_breakout(df, params)
    equity_pnl, equity_account, total_withdrawn, yearly_withdrawals, account_liquidations = apply_withdrawal_rule(
        equity_pnl_raw,
        threshold=params["balance"],  # 100k
        liquidation_level=params.get("liquidation_level")  # e.g., 90k or None
    )

    analyze_results(trades, equity_pnl, equity_account, df, total_withdrawn, yearly_withdrawals)

    # Save detailed outputs for later analysis in script directory
    out_trades = BASE_DIR / "backtest_trades.csv"
    out_pnl = BASE_DIR / "equity_pnl.csv"
    out_account = BASE_DIR / "equity_account.csv"
    trades.to_csv(out_trades, index=False)
    equity_pnl.to_csv(out_pnl, index=False)
    equity_account.to_csv(out_account, index=False)
    print(
        f"Results saved to {out_trades}, {out_pnl} and {out_account}"
    )
    # Save detailed outputs for later analysis
    trades.to_csv("backtest_trades.csv", index=False)
    equity_pnl.to_csv("equity_pnl.csv", index=False)
    equity_account.to_csv("equity_account.csv", index=False)
    print("Results saved to backtest_trades.csv, equity_pnl.csv and equity_account.csv")
