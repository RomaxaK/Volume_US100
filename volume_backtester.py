import pandas as pd
from math import floor
import matplotlib.pyplot as plt


PIP_VALUE_PER_LOT = 1.0
FEE_PER_LOT = 0.0


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

    balance = params["balance"]

    df = df.copy()
    df["vol_avg"] = df["tick_volume"].rolling(window=vol_lookback).mean().shift(1)

    equity_curve = [balance]
    equity_time = [df["time"].iloc[0]]
    trade_log = []

    total_sl = total_tp = total_partial = total_skips = 0
    liquidated_count = 0


    in_trade = False
    direction = None
    entry_price = entry_time = stop_price = trail_stop = partial_target = None
    partial_hit = False

    i = lookback
    while i < len(df) - 1:
        curr_time = df.loc[i, "time"]

        if not in_trade:
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
                        i += 1
                        continue
                    stop_distance = entry_price - stop_price
                    partial_target = entry_price + stop_distance
                else:
                    stop_price = recent_high
                    if stop_price <= entry_price:
                        in_trade = False
                        total_skips += 1
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
                    equity_curve.append(balance)
                elif direction == "short" and low <= partial_target:
                    partial_hit = True
                    profit_half = risk * 0.5
                    balance += profit_half
                    trail_stop = entry_price
                    print(
                        f"    ✅ Partial TP hit at {curr_time} - closed 50% at +1R. +${profit_half:.2f} realized, stop moved to {trail_stop:.2f}"
                    )
                    equity_time.append(curr_time)
                    equity_curve.append(balance)

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
                        "net_PnL": balance - params["balance"],
                    }
                )
                equity_time.append(exit_time)
                equity_curve.append(balance)

                if balance < liquidation_level:
                    liquidated_count += 1
                    print(
                        f"    💥 Account liquidated at {exit_time}. Resetting to {starting_balance:.2f}"
                    )
                    balance = starting_balance
                    equity_time.append(exit_time)
                    equity_curve.append(balance)


                in_trade = False
                direction = None
                entry_price = entry_time = None
                stop_price = trail_stop = partial_target = None
                partial_hit = False

        i += 1

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
                "net_PnL": balance - params["balance"],
            }
        )
        equity_time.append(final_time)
        equity_curve.append(balance)
        if balance < liquidation_level:
            liquidated_count += 1
            print(
                f"    💥 Account liquidated at {final_time}. Resetting to {starting_balance:.2f}"
            )
            balance = starting_balance
            equity_time.append(final_time)
            equity_curve.append(balance)

        print(
            f"    ⚠️ Trade open at end of data. Closing at {final_time} price {final_price:.2f}. Outcome: {outcome}"
        )
        print(f"    📊 Account balance after trade: ${balance:.2f}")

    trade_df = pd.DataFrame(trade_log)
    trade_df.attrs["total_skips"] = total_skips
    trade_df.attrs["total_sl"] = total_sl
    trade_df.attrs["total_tp"] = total_tp
    trade_df.attrs["total_partial"] = total_partial
    trade_df.attrs["liquidated_count"] = liquidated_count

    trade_df.attrs["risk"] = risk
    equity_df = pd.DataFrame({"time": equity_time, "balance": equity_curve})
    return trade_df, equity_df


def analyze_results(trade_log: pd.DataFrame, equity_curve: pd.DataFrame, df: pd.DataFrame) -> None:
    """Print a summary of results and plot the equity curve."""
    total_trades = len(trade_log)
    total_tp = (trade_log["outcome"] == "FULL_TP").sum()
    total_sl = (trade_log["outcome"] == "FULL_SL").sum()
    total_partial = (trade_log["outcome"] == "PARTIAL_SL").sum()
    total_skips = trade_log.attrs.get("total_skips", 0)
    liquidated_count = trade_log.attrs.get("liquidated_count", 0)

    wins = total_tp + total_partial
    win_rate = wins / total_trades * 100 if total_trades else 0.0
    net_profit = equity_curve["balance"].iloc[-1] - equity_curve["balance"].iloc[0]
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

    plt.figure(figsize=(8, 4))
    plt.plot(equity_curve["time"], equity_curve["balance"], label="Equity Curve")
    plt.title("NAS100 Breakout Strategy Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Account Balance")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_data("US100.cash_2024.csv")
    params = {
        "lookback": 20,
        "vol_lookback": 20,
        "vol_mult": 2.0,
        "risk": 2000.0,
        "balance": 100000.0,
        "liquidation_level": 90000.0,

    }
    trades, equity = backtest_volume_breakout(df, params)
    analyze_results(trades, equity, df)

