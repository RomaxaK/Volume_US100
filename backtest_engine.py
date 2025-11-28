import pandas as pd
from backtesting import Backtest, Strategy

# ================== FTMO CONFIG ==================
START_BALANCE       = 100_000      # starting account balance
LIQ_THRESHOLD       = 90_000       # liquidation if balance < this
WITHDRAW_MODE       = "biweekly"    # "none", "monthly", "biweekly", "target"
WITHDRAW_TARGET     = 2000        # only used if WITHDRAW_MODE == "target"
# ================== LOTs CONFIG ==================
MARGIN_PER_LOT_BUY  = 491.40       # USD margin required for 1.00 lot BUY
MARGIN_PER_LOT_SELL = 491.38       # USD margin required for 1.00 lot SELL
MIN_SL_DISTANCE     = 5.6          # minimum allowed SL distance (in price units)
COMMISSION_PER_LOT  = 0          # <-- USD commission per 1.00 lot per order
# ================== TIMEFRAME CONFIG ==================
ACTIVE_TIMEFRAMES = ["M5"]   # e.g. ["M1", "M5"], or ["M15"], or ["H1","H4"]

TIMEFRAME_FILES = {
    "M1":  r"US100M1.csv",
    "M3":  r"US100M3.csv",
    "M5":  r"US100M5.csv",
    "M15": r"US100M15.csv",
    "M30": r"US100M30.csv",
    "H1":  r"US100H1.csv",
    "H4":  r"US100H4.csv",
    "D1":  r"US100D1.csv",
}
# ======================================================



class BOSOnlyStrategy(Strategy):
    """
    Very simple BOS-only strategy on a single timeframe (M15).

    - Long: close breaks above previous N-bar high  -> SL at swing low, TP = RR * SL distance.
    - Short: close breaks below previous N-bar low -> SL at swing high, TP = RR * SL distance.
    """

    # Configurable parameters
    risk_per_trade_pct = 0.01   # 1% of equity per trade
    rr = 2.0                    # risk:reward (2R)
    bos_lookback = 10           # number of candles to look back for BOS

    def init(self):
        # Nothing fancy here; we operate directly on self.data
        pass

    # Convenience: current price
    @property
    def price(self) -> float:
        return float(self.data.Close[-1])

    def _calc_position_size(self, sl_price: float, is_long: bool) -> float:
        """
        Position size in LOTS, constrained by:
        - minimum SL distance (MIN_SL_DISTANCE)
        - risk_per_trade_pct of current equity
        - margin per 1.00 lot (different for buy/sell)
        Backtesting.py requires:
          - 0 < size < 1  => fraction of equity
          - size >= 1 and integer => number of units
        We use integer LOTS; trades needing <1 lot are skipped.
        """
        equity = float(self.equity)
        price = self.price
        distance = abs(price - sl_price)

        # Enforce minimum SL and positive equity
        if distance < MIN_SL_DISTANCE or equity <= 0:
            return 0.0

        # Assume 1 lot moves 1 USD per 1.0 price unit (adjust if needed).
        value_per_point_per_lot = 1.0

        # Risk per trade in USD
        risk_amount = START_BALANCE * self.risk_per_trade_pct

        # Ideal lots from risk and SL distance:
        # risk = distance * value_per_point_per_lot * lots
        lots_from_risk = risk_amount / (distance * value_per_point_per_lot)
        if lots_from_risk <= 0:
            return 0.0

        # Approximate free margin by equity (single position at a time)
        margin_per_lot = MARGIN_PER_LOT_BUY if is_long else MARGIN_PER_LOT_SELL
        max_lots_by_margin = equity / margin_per_lot if margin_per_lot > 0 else lots_from_risk

        # Raw lots limited by margin cap
        lots_raw = min(lots_from_risk, max_lots_by_margin)

        # If we’d need less than 1 lot, skip this trade (cannot represent fractional lots as units)
        if lots_raw < 1.0:
            return 0.0

        # Backtesting requires integer units when >= 1
        lots_int = int(round(lots_raw))
        if lots_int <= 0:
            return 0.0

        return float(lots_int)

    def _enter_long(self):
        close = self.data.Close
        low = self.data.Low

        if len(close) < self.bos_lookback + 1:
            return

        entry = float(close[-1])

        # Swing low in the lookback window *before* current bar
        swing_low = float(low[-(self.bos_lookback + 1):-1].min())

        # Sanity: SL must be below entry
        if swing_low >= entry:
            return

        sl = swing_low
        sl_dist = entry - sl
        tp = entry + sl_dist * self.rr

        # Approximate fill price:
        # trade_on_close = False -> enter at next bar open
        # Here we approximate with current bar open; good enough for constraints.
        approx_fill = float(self.data.Open[-1])

        # Long constraint: SL < fill < TP
        if not (sl < approx_fill < tp):
            return

        size = self._calc_position_size(sl, is_long=True)
        if size <= 0:
            return  # includes: SL < MIN_SL_DISTANCE, or no margin, etc.

        self.buy(size=size, sl=sl, tp=tp)

    def _enter_short(self):
        close = self.data.Close
        high = self.data.High

        if len(close) < self.bos_lookback + 1:
            return

        entry = float(close[-1])

        # Swing high in the lookback window *before* current bar
        swing_high = float(high[-(self.bos_lookback + 1):-1].max())

        # Sanity: SL must be above entry
        if swing_high <= entry:
            return

        sl = swing_high
        sl_dist = sl - entry
        tp = entry - sl_dist * self.rr

        # Approximate fill price for constraints (see comment in _enter_long)
        approx_fill = float(self.data.Open[-1])

        # IMPORTANT: Backtesting's short constraint:
        # TP < LIMIT/entry < SL
        # If this is not true (e.g. huge gap through TP), we SKIP the trade
        if not (tp < approx_fill < sl):
            return

        size = self._calc_position_size(sl, is_long=False)
        if size <= 0:
            return  # includes: SL < MIN_SL_DISTANCE, or no margin, etc.

        self.sell(size=size, sl=sl, tp=tp)

    def next(self):
        # Don't stack positions; one trade at a time
        if self.position:
            return

        high = self.data.High
        low = self.data.Low
        close = self.data.Close

        if len(close) < self.bos_lookback + 1:
            return

        last_close = float(close[-1])
        prev_high = float(high[-(self.bos_lookback + 1):-1].max())
        prev_low = float(low[-(self.bos_lookback + 1):-1].min())

        bullish_bos = last_close > prev_high
        bearish_bos = last_close < prev_low

        if bullish_bos:
            self._enter_long()
        elif bearish_bos:
            self._enter_short()


def load_mt5_csv(path: str) -> pd.DataFrame:
    """
    Load MT5 CSV for any timeframe:
    Header: Time;Open;High;Low;Close;Tick Volume
    Time format: YYYY.MM.DD HH:MM
    """
    df = pd.read_csv(
        path,
        sep=";",          # VERY IMPORTANT for your file
        encoding="utf-8"
    )

    df["Time"] = pd.to_datetime(df["Time"], format="%Y.%m.%d %H:%M")

    df["Time"] = (
        df["Time"]
        .dt.tz_localize("Etc/GMT-2")   # <-- server/broker timezone
        .dt.tz_convert("UTC")          # convert to UTC
        .dt.tz_localize(None)          # make it naive UTC
    )

    df["Date"] = df["Time"]

    df = df.rename(
        columns={
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Tick Volume": "Volume",
        }
    )

    df = df.set_index("Date")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    return df


def apply_ftmo_equity(stats: pd.Series) -> pd.Series:
    """
    Build FTMO-style equity curve using FIXED RISK per trade.
    Backtesting is only used to decide TP/SL and timestamps.

    - Risk per trade in USD = START_BALANCE * risk_per_trade_pct (constant)
    - TP  => +rr * risk_usd
    - SL  => -1 * risk_usd
    - Then apply withdrawals (monthly / biweekly / target) and liquidation.
    """

    trades = stats._trades.copy()
    if trades.empty:
        ftmo_eq = pd.Series(
            [START_BALANCE],
            index=[stats._equity_curve.index[0]],
            name="FTMO_Equity",
        )
        stats.ftmo_equity_curve = ftmo_eq
        stats.ftmo_withdrawals = []
        stats.ftmo_liq_dates = []
        stats.balance_after_each_trade = []
        stats.balance_by_trade_index = {}
        stats["FTMO Equity Final [$]"] = START_BALANCE
        stats["FTMO Total Withdrawn [$]"] = 0.0
        stats["FTMO Liquidations"] = 0
        stats["FTMO Withdrawals Count"] = 0
        return stats

    # Sort trades by ExitTime and remember original index
    trades = trades.sort_values("ExitTime")
    risk_usd = START_BALANCE * BOSOnlyStrategy.risk_per_trade_pct
    rr = BOSOnlyStrategy.rr

    bal = float(START_BALANCE)
    total_withdrawn = 0.0
    withdraw_events = []
    liq_dates = []
    liq_count = 0

    first_exit = trades["ExitTime"].iloc[0]
    prev_date = first_exit
    last_payout = first_exit
    blocked = False
    block_year = first_exit.year
    block_month = first_exit.month

    ftmo_index = []
    ftmo_values = []
    balance_after_each_trade = []
    balance_by_trade_index = {}

    for idx, (trade_idx, tr) in enumerate(trades.iterrows()):
        d = tr["ExitTime"]

        # ===== withdrawals BEFORE applying this trade =====
        if WITHDRAW_MODE == "monthly" and d.month != prev_date.month:
            if bal > START_BALANCE:
                amt = bal - START_BALANCE
                total_withdrawn += amt
                withdraw_events.append((d, amt, "monthly"))
                bal = START_BALANCE

        if WITHDRAW_MODE == "biweekly" and (d - last_payout).days >= 14:
            if bal > START_BALANCE:
                amt = bal - START_BALANCE
                total_withdrawn += amt
                withdraw_events.append((d, amt, "biweekly"))
                bal = START_BALANCE
                last_payout = d

        if WITHDRAW_MODE == "target":
            # unblock at new month
            if blocked and (d.year != block_year or d.month != block_month):
                blocked = False
            # if blocked -> no trading effect on balance
            if blocked:
                ftmo_index.append(d)
                ftmo_values.append(bal)
                prev_date = d
                balance_after_each_trade.append((d, bal))
                balance_by_trade_index[trade_idx] = bal
                continue

        # ===== determine outcome (TP / SL) from prices =====
        entry = float(tr["EntryPrice"])
        exit_price = float(tr["ExitPrice"])
        size = float(tr["Size"])
        direction = "LONG" if size > 0 else "SHORT"

        if direction == "LONG":
            win = exit_price >= entry
        else:
            win = exit_price <= entry

        # ===== apply FIXED-RISK PnL =====
        if win:
            bal += risk_usd * rr
        else:
            bal -= risk_usd

        # ===== target-profit withdrawal AFTER trade =====
        if WITHDRAW_MODE == "target" and not blocked:
            if bal >= START_BALANCE + WITHDRAW_TARGET:
                amt = bal - START_BALANCE
                total_withdrawn += amt
                withdraw_events.append((d, amt, "target"))
                bal = START_BALANCE
                blocked = True
                block_year, block_month = d.year, d.month

        # ===== liquidation =====
        if bal < LIQ_THRESHOLD:
            liq_count += 1
            liq_dates.append(d)
            bal = START_BALANCE

        # record equity at this trade's exit time
        ftmo_index.append(d)
        ftmo_values.append(bal)

        balance_after_each_trade.append((d, bal))
        balance_by_trade_index[trade_idx] = bal

        prev_date = d

    ftmo_eq = pd.Series(ftmo_values, index=ftmo_index, name="FTMO_Equity")

    stats.ftmo_equity_curve = ftmo_eq
    stats.ftmo_withdrawals = withdraw_events
    stats.ftmo_liq_dates = liq_dates
    stats.balance_after_each_trade = balance_after_each_trade
    stats.balance_by_trade_index = balance_by_trade_index

    stats["FTMO Equity Final [$]"] = ftmo_eq.iloc[-1]
    stats["FTMO Total Withdrawn [$]"] = total_withdrawn
    stats["FTMO Liquidations"] = liq_count
    stats["FTMO Withdrawals Count"] = len(withdraw_events)

    return stats

def export_trades_to_csv(stats: pd.Series, filename: str = "Trades.csv") -> None:
    """
    Export each trade into a CSV with columns:
    date, entry_price, sl_price, tp_price, entry_time, exit_time,
    direction, outcome (TP/SL), RR, risk_usd, account_balance_after
    """
    trades = stats._trades.copy()

    # Default values so print_reports() has something to read
    stats.csv_exported = None
    stats.csv_exported_trades = 0

    if trades.empty:
        print("\n[trade-log] No trades -> no CSV created.")
        return

    # Mapping from original trade index -> FTMO balance after that trade
    balance_map = getattr(stats, "balance_by_trade_index", None)

    rows = []
    rr = BOSOnlyStrategy.rr
    risk_usd_const = START_BALANCE * BOSOnlyStrategy.risk_per_trade_pct

    for trade_idx, tr in trades.iterrows():
        entry_time = tr["EntryTime"]
        exit_time = tr["ExitTime"]
        entry = float(tr["EntryPrice"])
        exit_price = float(tr["ExitPrice"])
        size = float(tr["Size"])

        direction = "LONG" if size > 0 else "SHORT"

        if direction == "LONG":
            if exit_price < entry:
                sl = exit_price
                tp = entry + (entry - sl) * rr
                outcome = "SL"
            else:
                tp = exit_price
                sl = entry - (tp - entry) / rr
                outcome = "TP"
        else:
            if exit_price > entry:
                sl = exit_price
                tp = entry - (sl - entry) * rr
                outcome = "SL"
            else:
                tp = exit_price
                sl = entry + (entry - tp) / rr
                outcome = "TP"

        R = rr if outcome == "TP" else -1.0

        if balance_map is not None and trade_idx in balance_map:
            balance_after = balance_map[trade_idx]
        else:
            if hasattr(stats, "balance_after_each_trade") and stats.balance_after_each_trade:
                balance_after = stats.balance_after_each_trade[-1][1]
            else:
                balance_after = START_BALANCE

        rows.append(
            {
                "date": entry_time.date(),
                "entry_price": entry,
                "sl_price": sl,
                "tp_price": tp,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "direction": direction,
                "outcome": outcome,
                "RR": R,
                "risk_usd": risk_usd_const,
                "account_balance_after": balance_after,
            }
        )

    trades_df = pd.DataFrame(rows)
    trades_df.to_csv(filename, index=False)

    # 🔹 tell print_reports() that CSV was created
    stats.csv_exported = filename
    stats.csv_exported_trades = len(trades_df)

    print(f"\n[trade-log] Saved {filename} with {len(trades_df)} trades.")


def commission_per_lot(size: float, price: float) -> float:
    """
    Backtesting.py commission callback.

    Called as commission(size, price).
    We charge COMMISSION_PER_LOT USD per 1.0 lot per order.
    So a full round-trip (entry + exit) costs:
        abs(size) * COMMISSION_PER_LOT * 2
    """
    return abs(size) * COMMISSION_PER_LOT

def print_reports(stats: pd.Series) -> None:
    """
    Print all summary blocks:
    - Header (timeframe, data CSV, export CSV)
    - Backtesting.py original stats
    - FTMO summary
    - Withdrawals
    - Liquidations
    - Trade statistics
    """


    tf = getattr(stats, "timeframe", None)
    data_path = getattr(stats, "data_path", None)
    csv_exported = getattr(stats, "csv_exported", None)
    csv_trades = getattr(stats, "csv_exported_trades", 0)
    for tf in ACTIVE_TIMEFRAMES:
        if tf not in TIMEFRAME_FILES:
            raise ValueError(
                f"ACTIVE_TIMEFRAMES contains '{tf}' which is not in TIMEFRAME_FILES. "
                f"Available: {list(TIMEFRAME_FILES.keys())}"
            )
    print("\n======================================")
    if tf is not None:
        print(f" TIMEFRAME: {tf}")
    if data_path is not None:
        print(f" DATA CSV: {data_path}")
    if csv_exported is not None:
        print(f" TRADES CSV: {csv_exported} ({csv_trades} trades)")
    else:
        print(" TRADES CSV: no trades, nothing exported")
    print("======================================")

    # ===== BACKTESTING.PY ORIGINAL STATS =====
    print("\n===== BACKTESTING.PY ORIGINAL STATS =====")
    print(stats)
    # ===== FTMO SUMMARY =====
    print("\n===== FTMO SUMMARY =====")
    print(f"Final FTMO Equity:      ${stats['FTMO Equity Final [$]']:.2f}")
    print(f"Total Withdrawn:        ${stats['FTMO Total Withdrawn [$]']:.2f}")
    print(f"Withdrawals Count:      {stats['FTMO Withdrawals Count']}")
    print(f"Liquidations Count:     {stats['FTMO Liquidations']}")

    # ===== WITHDRAWALS (DETAIL) =====
    print("\n===== WITHDRAWALS (DETAIL) =====")
    if stats.ftmo_withdrawals:
        for d, amt, mode in stats.ftmo_withdrawals:
            print(f"{d.date()} | mode={mode:<8} | amount=${amt:,.2f}")
        print(f"-- TOTAL WITHDRAWN: ${stats['FTMO Total Withdrawn [$]']:.2f}")
    else:
        print("No withdrawals.")

    # ===== LIQUIDATIONS (DETAIL) =====
    print("\n===== LIQUIDATIONS (DETAIL) =====")
    if stats.ftmo_liq_dates:
        print(f"Total liquidations: {stats['FTMO Liquidations']}")
        for d in stats.ftmo_liq_dates:
            print(f"- {d}")
    else:
        print("No liquidations.")

    # ===== TRADE STATS =====
    print("\n===== TRADE STATS =====")
    trades_df = stats._trades.copy()

    if trades_df.empty:
        print("No trades.")
    else:
        total_trades = len(trades_df)
        tp_count = (trades_df["PnL"] > 0).sum()
        sl_count = (trades_df["PnL"] < 0).sum()
        winrate = (tp_count / total_trades) * 100 if total_trades else 0.0

        print(f"Total trades: {total_trades}")
        print(f"TP trades:    {tp_count}")
        print(f"SL trades:    {sl_count}")
        print(f"Winrate:      {winrate:.2f}%")


if __name__ == "__main__":

    for tf in ACTIVE_TIMEFRAMES:
        data_path = TIMEFRAME_FILES[tf]
        df = load_mt5_csv(data_path)

        bt = Backtest(
            df,
            BOSOnlyStrategy,
            cash=START_BALANCE,
            commission=commission_per_lot,
            trade_on_close=False,
            exclusive_orders=True,
        )

        stats = bt.run()
        stats = apply_ftmo_equity(stats)

        # tag stats so print_reports can show correct header
        stats.timeframe = tf
        stats.data_path = data_path

        csv_name = f"Trades_{tf}.csv" if len(ACTIVE_TIMEFRAMES) > 1 else "Trades.csv"
        export_trades_to_csv(stats, filename=csv_name)
        print_reports(stats)

        bt.plot()
