"""Herramienta monoarchivo para automatizar estrategias de inversión.

Este script proporciona utilidades básicas para:
- Descargar datos de mercado (Yahoo Finance y, opcionalmente, CCXT).
- Ejecutar una estrategia simple de cruce de medias móviles.
- Simular operaciones en un broker *paper* o enviar órdenes a Alpaca.
- Guardar y cargar configuraciones en formato YAML.
- Realizar backtests y generar una curva de capital.

Advertencia: el código tiene fines educativos y debe ser auditado antes de
utilizarse con dinero real.
"""

from __future__ import annotations

import json
import math
import os
import sys
from copy import deepcopy
from csv import DictWriter
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Protocol

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover - dependencias externas
    print("Falta instalar pandas y/o numpy. Ejecuta: pip install pandas numpy", file=sys.stderr)
    raise

try:
    import yfinance as yf
except Exception as exc:  # pragma: no cover - dependencias externas
    print("Falta instalar yfinance. Ejecuta: pip install yfinance", file=sys.stderr)
    raise

try:
    import typer
except Exception as exc:  # pragma: no cover - dependencias externas
    print("Falta instalar Typer. Ejecuta: pip install typer", file=sys.stderr)
    raise

EPSILON = 1e-8
ConfigDict = dict[str, Any]


# ============================================================
# IMPORTS OPCIONALES
# ============================================================

def _safe_import_ccxt():  # pragma: no cover - import opcional
    try:
        import ccxt  # type: ignore

        return ccxt
    except Exception:
        return None


def _safe_import_yaml():  # pragma: no cover - import opcional
    try:
        import yaml  # type: ignore

        return yaml
    except Exception:
        return None


def _safe_import_dotenv():  # pragma: no cover - import opcional
    try:
        from dotenv import load_dotenv  # type: ignore

        return load_dotenv
    except Exception:
        return None


def _safe_import_matplotlib():  # pragma: no cover - import opcional
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


def _safe_import_apscheduler():  # pragma: no cover - import opcional
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler  # type: ignore

        return BlockingScheduler
    except Exception:
        return None


# ============================================================
# CONFIGURACIÓN POR DEFECTO
# ============================================================

DEFAULT_CONFIG: ConfigDict = {
    "portfolio": {
        "cash_start": 10_000.0,
        "tickers": ["SPY", "QQQ"],
        "crypto": ["BTC/USDT:BINANCE"],
    },
    "strategy": {"name": "sma_cross", "params": {"fast": 20, "slow": 50}},
    "risk": {"risk_per_trade": 0.01, "stop_pct": 0.05},
    "data": {"interval": "1d", "lookback_days": 400},
    "execution": {"broker": "paper"},
    "reporting": {"output_dir": "reports", "state_file": "reports/state.json"},
}

_load_dotenv = _safe_import_dotenv()
if _load_dotenv:  # pragma: no cover - import opcional
    _load_dotenv()


# ============================================================
# UTILIDADES DE CONFIGURACIÓN Y ESTADO
# ============================================================

def deep_merge(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    """Combina dos diccionarios recursivamente."""

    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = value
    return result


def read_config(path: Optional[str]) -> ConfigDict:
    """Carga la configuración desde YAML si se proporciona ruta; si no, usa la predeterminada."""

    if not path:
        return deepcopy(DEFAULT_CONFIG)

    yaml = _safe_import_yaml()
    if not yaml:
        typer.secho("PyYAML no está instalado. Ejecuta: pip install pyyaml", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    cfg_path = Path(path)
    if not cfg_path.exists():
        typer.secho(f"No existe el archivo de configuración: {path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        typer.secho("El archivo de configuración debe contener un diccionario.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    merged = deep_merge(deepcopy(DEFAULT_CONFIG), data)
    return merged


def write_config_file(config: ConfigDict, path: str) -> None:
    """Guarda la configuración en un archivo YAML."""

    yaml = _safe_import_yaml()
    if not yaml:
        typer.secho("PyYAML no está instalado. Ejecuta: pip install pyyaml", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    cfg_path = Path(path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    typer.secho(f"Configuración guardada en {path}", fg=typer.colors.GREEN)


def load_state(path: str, cash_start: float) -> ConfigDict:
    """Lee el estado persistido del broker paper."""

    state_path = Path(path)
    if not state_path.exists():
        return {"cash": float(cash_start), "positions": {}}

    try:
        raw_data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {"cash": float(cash_start), "positions": {}}

    cash = float(raw_data.get("cash", cash_start))
    positions_raw = raw_data.get("positions", {})
    positions: dict[str, float] = {}

    if isinstance(positions_raw, dict):
        for symbol, qty in positions_raw.items():
            try:
                qty_f = float(qty)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(qty_f) or abs(qty_f) <= EPSILON:
                continue
            positions[str(symbol)] = qty_f

    return {"cash": cash, "positions": positions}


def save_state(state: ConfigDict, path: str) -> None:
    """Persistencia sencilla del broker paper."""

    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    clean_positions: dict[str, float] = {}
    for symbol, qty in state.get("positions", {}).items():
        try:
            qty_f = float(qty)
        except (TypeError, ValueError):
            continue
        if math.isfinite(qty_f) and abs(qty_f) > EPSILON:
            clean_positions[str(symbol)] = qty_f

    payload = {"cash": float(state.get("cash", 0.0)), "positions": clean_positions}
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# DATOS DE MERCADO
# ============================================================

def _ensure_close_column(df: pd.DataFrame, source: str) -> None:
    if "close" not in df.columns:
        raise ValueError(f"Los datos obtenidos desde {source} no contienen columna 'close'.")


def get_history_yf(ticker: str, interval: str = "1d", lookback_days: int = 400) -> pd.DataFrame:
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days + 10)
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        progress=False,
        auto_adjust=False,
    )
    if df is None or df.empty:
        raise ValueError(f"No hay datos para {ticker} (yfinance).")

    df = df.rename(columns=str.lower)
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tzinfo", None) is not None:
        df.index = df.index.tz_convert(None)

    _ensure_close_column(df, "yfinance")
    return df


def parse_ccxt_symbol(symbol_with_ex: str) -> tuple[str, str]:
    parts = symbol_with_ex.split(":")
    if len(parts) != 2:
        raise ValueError("El símbolo de CCXT debe tener el formato 'PAR:EXCHANGE'.")
    symbol, exchange = parts
    symbol = symbol.strip()
    exchange = exchange.strip().lower()
    if not symbol or not exchange:
        raise ValueError("El símbolo o el exchange de CCXT no pueden estar vacíos.")
    return symbol, exchange


def get_history_ccxt(symbol_with_ex: str, timeframe: str = "1d", lookback_days: int = 400) -> pd.DataFrame:
    ccxt = _safe_import_ccxt()
    if not ccxt:
        raise RuntimeError("CCXT no está instalado. Ejecuta: pip install ccxt")

    symbol, exchange_name = parse_ccxt_symbol(symbol_with_ex)
    if not hasattr(ccxt, exchange_name):
        raise ValueError(f"Exchange CCXT no soportado: {exchange_name}")

    exchange_cls = getattr(ccxt, exchange_name)
    exchange = exchange_cls({"enableRateLimit": True})
    since_ms = int((datetime.utcnow() - timedelta(days=lookback_days)).timestamp() * 1000)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms)
    if not ohlcv:
        raise ValueError(f"No hay datos para {symbol_with_ex} (CCXT).")

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    df = df.set_index("timestamp")
    _ensure_close_column(df, "ccxt")
    return df


# ============================================================
# ESTRATEGIA
# ============================================================


class Strategy(Protocol):
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Devuelve una serie de señales (1 = largo, 0 = fuera)."""


class SmaCross:
    def __init__(self, fast: int = 20, slow: int = 50) -> None:
        if fast >= slow:
            raise ValueError("El parámetro 'fast' debe ser menor que 'slow'.")
        self.fast = fast
        self.slow = slow

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        sma_fast = close.rolling(self.fast, min_periods=self.fast).mean()
        sma_slow = close.rolling(self.slow, min_periods=self.slow).mean()
        signals = (sma_fast > sma_slow).astype(int)
        return signals.fillna(0)


def build_strategy(config: ConfigDict) -> Strategy:
    name = config["strategy"]["name"].lower()
    params = config["strategy"].get("params", {})
    if name == "sma_cross":
        return SmaCross(**params)
    raise ValueError(f"Estrategia no soportada: {name}")


# ============================================================
# RIESGO / SIZING
# ============================================================

def position_size(
    balance: float,
    price: float,
    risk_per_trade: float = 0.01,
    stop_pct: float = 0.05,
) -> int:
    if price <= 0 or stop_pct <= 0 or risk_per_trade <= 0:
        return 0
    if not (math.isfinite(balance) and math.isfinite(price)):
        return 0

    units = (balance * risk_per_trade) / (price * stop_pct)
    if not math.isfinite(units):
        return 0
    return max(int(math.floor(units)), 0)


# ============================================================
# BROKERS
# ============================================================


class BrokerBase(Protocol):
    def get_cash(self) -> float:
        ...

    def get_position(self, symbol: str) -> float:
        ...

    def market_buy(self, symbol: str, qty: float, price: float | None = None) -> dict:
        ...

    def market_sell(self, symbol: str, qty: float, price: float | None = None) -> dict:
        ...


class PaperBroker:
    def __init__(self, cash_start: float, positions: Optional[dict[str, float]] = None) -> None:
        self.cash = float(cash_start)
        self.positions: dict[str, float] = {**(positions or {})}

    def get_cash(self) -> float:
        return self.cash

    def get_position(self, symbol: str) -> float:
        return float(self.positions.get(symbol, 0.0))

    def market_buy(self, symbol: str, qty: float, price: float | None = None) -> dict:
        qty = float(qty)
        if qty <= 0:
            return {"status": "rejected", "reason": "qty<=0"}
        if price is None or price <= 0:
            return {"status": "rejected", "reason": "price<=0"}

        cost = qty * price
        if cost > self.cash + EPSILON:
            return {"status": "rejected", "reason": "insufficient_cash"}

        self.cash -= cost
        self.positions[symbol] = self.get_position(symbol) + qty
        return {"status": "filled", "side": "buy", "symbol": symbol, "qty": qty, "price": price}

    def market_sell(self, symbol: str, qty: float, price: float | None = None) -> dict:
        qty = float(qty)
        if qty <= 0:
            return {"status": "rejected", "reason": "qty<=0"}
        if price is None or price <= 0:
            return {"status": "rejected", "reason": "price<=0"}

        position_qty = self.get_position(symbol)
        if qty - position_qty > EPSILON:
            return {"status": "rejected", "reason": "insufficient_position"}

        proceeds = qty * price
        self.cash += proceeds
        new_qty = position_qty - qty
        if abs(new_qty) <= EPSILON:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = new_qty
        return {"status": "filled", "side": "sell", "symbol": symbol, "qty": qty, "price": price}


class AlpacaBroker:
    """Broker opcional que interactúa con la API de Alpaca."""

    def __init__(self) -> None:  # pragma: no cover - requiere credenciales
        try:
            import alpaca_trade_api as tradeapi  # type: ignore
        except Exception as exc:
            raise RuntimeError("alpaca-trade-api no instalado. Ejecuta: pip install alpaca-trade-api") from exc

        api_key = os.getenv("ALPACA_API_KEY_ID")
        api_secret = os.getenv("ALPACA_API_SECRET_KEY")
        base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        if not api_key or not api_secret:
            raise RuntimeError("Faltan credenciales Alpaca en variables de entorno.")

        self.api = tradeapi.REST(api_key, api_secret, base_url=base_url, api_version="v2")

    def get_cash(self) -> float:
        account = self.api.get_account()
        return float(account.cash)

    def get_position(self, symbol: str) -> float:
        try:
            position = self.api.get_position(symbol)
            return float(position.qty)
        except Exception:
            return 0.0

    def market_buy(self, symbol: str, qty: float, price: float | None = None) -> dict:
        order = self.api.submit_order(symbol=symbol, qty=str(qty), side="buy", type="market", time_in_force="day")
        return {
            "status": "submitted",
            "id": getattr(order, "id", None),
            "side": "buy",
            "symbol": symbol,
            "qty": qty,
        }

    def market_sell(self, symbol: str, qty: float, price: float | None = None) -> dict:
        order = self.api.submit_order(symbol=symbol, qty=str(qty), side="sell", type="market", time_in_force="day")
        return {
            "status": "submitted",
            "id": getattr(order, "id", None),
            "side": "sell",
            "symbol": symbol,
            "qty": qty,
        }


def build_broker(config: ConfigDict, state: ConfigDict) -> BrokerBase:
    broker_name = config["execution"].get("broker", "paper").lower()
    if broker_name == "paper":
        return PaperBroker(cash_start=state["cash"], positions=state.get("positions", {}))
    if broker_name == "alpaca":
        return AlpacaBroker()
    raise ValueError(f"Broker no soportado: {broker_name}")


# ============================================================
# OPERATIVA
# ============================================================


@dataclass
class Trade:
    symbol: str
    side: str
    qty: float
    price: float
    timestamp: str


def evaluate_last_signal_change(
    symbol: str,
    df: pd.DataFrame,
    signal: pd.Series,
    broker: BrokerBase,
    risk_cfg: ConfigDict,
) -> list[Trade]:
    trades: list[Trade] = []
    if len(signal) < 2:
        return trades

    last_price = float(df["close"].iloc[-1])
    if not math.isfinite(last_price):
        return trades

    last_sig = int(signal.iloc[-1])
    prev_sig = int(signal.iloc[-2])
    last_ts = pd.Timestamp(df.index[-1]).to_pydatetime().isoformat()

    if last_sig == 1 and prev_sig == 0:
        balance = broker.get_cash()
        qty = position_size(
            balance,
            last_price,
            risk_cfg.get("risk_per_trade", 0.01),
            risk_cfg.get("stop_pct", 0.05),
        )
        if qty > 0:
            response = broker.market_buy(symbol, qty, price=last_price)
            if response.get("status") in {"filled", "submitted"}:
                trades.append(Trade(symbol, "buy", qty, last_price, last_ts))
    elif last_sig == 0 and prev_sig == 1:
        position_qty = broker.get_position(symbol)
        if position_qty > 0:
            response = broker.market_sell(symbol, position_qty, price=last_price)
            if response.get("status") in {"filled", "submitted"}:
                trades.append(Trade(symbol, "sell", position_qty, last_price, last_ts))

    return trades


def append_trades_csv(trades: list[Trade], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = DictWriter(fh, fieldnames=["timestamp", "symbol", "side", "qty", "price"])
        if not file_exists:
            writer.writeheader()
        for trade in trades:
            writer.writerow(asdict(trade))


def calculate_equity_snapshot(broker: BrokerBase, prices: dict[str, float]) -> dict[str, Any]:
    cash = broker.get_cash()
    equity = cash
    detailed_positions: dict[str, dict[str, float]] = {}

    for symbol, price in prices.items():
        qty = broker.get_position(symbol)
        if abs(qty) <= EPSILON:
            continue
        value = qty * price
        equity += value
        detailed_positions[symbol] = {"qty": qty, "price": price, "value": value}

    return {
        "date": datetime.utcnow(),
        "equity": equity,
        "cash": cash,
        "positions": detailed_positions,
    }


# ============================================================
# BACKTEST
# ============================================================


def backtest_sma_cross(prices: pd.DataFrame, fast: int, slow: int, cash_start: float) -> pd.DataFrame:
    _ensure_close_column(prices, "backtest")
    close = prices["close"].astype(float)
    sma_fast = close.rolling(fast, min_periods=fast).mean()
    sma_slow = close.rolling(slow, min_periods=slow).mean()
    signals = (sma_fast > sma_slow).astype(int).fillna(0)

    position = 0
    cash = cash_start
    qty = 0
    equity: list[float] = []

    for timestamp, price in close.items():
        target = int(signals.loc[timestamp])
        if target == 1 and position == 0:
            qty = int(cash // price)
            cash -= qty * price
            position = 1
        elif target == 0 and position == 1:
            cash += qty * price
            qty = 0
            position = 0
        equity.append(cash + qty * price)

    return pd.DataFrame({"date": close.index, "equity": equity})


# ============================================================
# REPORTING
# ============================================================


def plot_equity_curve(equity_csv: str, out_path: str) -> None:
    plt = _safe_import_matplotlib()
    if not plt:
        typer.secho("Matplotlib no está instalado. Ejecuta: pip install matplotlib", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    df = pd.read_csv(equity_csv, parse_dates=["date"]).set_index("date")
    plt.figure()
    df["equity"].plot()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.title("Curva de capital")
    plt.xlabel("Fecha")
    plt.ylabel("Capital")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    typer.secho(f"Gráfico guardado en {out_path}", fg=typer.colors.GREEN)


# ============================================================
# RUTINAS PRINCIPALES
# ============================================================


def run_once_core(config: ConfigDict) -> None:
    output_dir = Path(config["reporting"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    state_file = config["reporting"]["state_file"]
    state = load_state(state_file, config["portfolio"]["cash_start"])
    broker = build_broker(config, state)
    strategy = build_strategy(config)
    risk_cfg = config["risk"]
    interval = config["data"]["interval"]
    lookback = config["data"]["lookback_days"]

    executed_trades: list[Trade] = []
    latest_prices: dict[str, float] = {}

    for ticker in config["portfolio"].get("tickers", []):
        try:
            df = get_history_yf(ticker, interval=interval, lookback_days=lookback)
        except Exception as exc:
            typer.secho(f"[WARN] No se pudo descargar datos de {ticker}: {exc}", fg=typer.colors.YELLOW)
            continue

        signals = strategy.generate_signals(df)
        trades = evaluate_last_signal_change(ticker, df, signals, broker, risk_cfg)
        if trades:
            append_trades_csv(trades, output_dir / f"trades_{ticker}.csv")
            executed_trades.extend(trades)

        latest_prices[ticker] = float(df["close"].iloc[-1])

    for pair in config["portfolio"].get("crypto", []):
        try:
            df = get_history_ccxt(pair, timeframe="1d", lookback_days=lookback)
            latest_prices[pair] = float(df["close"].iloc[-1])
        except Exception as exc:
            typer.secho(f"[WARN] CCXT/cripto omitido '{pair}': {exc}", fg=typer.colors.YELLOW)

    if isinstance(broker, PaperBroker):
        save_state({"cash": broker.get_cash(), "positions": broker.positions}, state_file)

    if not latest_prices:
        typer.secho("No se generaron registros de equity.", fg=typer.colors.YELLOW)
        return

    snapshot = calculate_equity_snapshot(broker, latest_prices)
    equity_path = output_dir / "equity_curve.csv"
    snapshot_df = pd.DataFrame([snapshot])

    if equity_path.exists():
        existing = pd.read_csv(equity_path, parse_dates=["date"])
        combined = pd.concat([existing, snapshot_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"]).sort_values("date")
    else:
        combined = snapshot_df

    combined.to_csv(equity_path, index=False)
    typer.secho(f"Curva de equity actualizada en {equity_path}", fg=typer.colors.GREEN)

    if executed_trades:
        typer.secho(f"Órdenes ejecutadas: {len(executed_trades)}", fg=typer.colors.GREEN)
    else:
        typer.secho("Sin nuevas operaciones en esta ejecución.", fg=typer.colors.BLUE)


def backtest_core(config: ConfigDict) -> Path:
    tickers = config["portfolio"].get("tickers", [])
    if not tickers:
        typer.secho("No hay tickers definidos en la configuración.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    prices = get_history_yf(tickers[0], interval=config["data"]["interval"], lookback_days=config["data"]["lookback_days"])
    params = config["strategy"].get("params", {})
    fast = int(params.get("fast", 20))
    slow = int(params.get("slow", 50))

    equity_df = backtest_sma_cross(prices, fast=fast, slow=slow, cash_start=config["portfolio"]["cash_start"])
    output_dir = Path(config["reporting"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "equity_curve_backtest.csv"
    equity_df.to_csv(out_path, index=False)
    return out_path


# ============================================================
# CLI (Typer)
# ============================================================

app = typer.Typer(help="Automatización de inversiones (paper por defecto).")


@app.command()
def show_config() -> None:
    """Muestra la configuración por defecto."""

    typer.echo(json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2))


@app.command("save-config")
def cli_save_config(path: str = typer.Option(..., help="Ruta destino YAML")) -> None:
    """Guarda la configuración por defecto en YAML."""

    write_config_file(DEFAULT_CONFIG, path)


@app.command("load-config")
def cli_load_config(path: str = typer.Option(..., help="Ruta YAML para cargar configuración y ejecutar backtest")) -> None:
    """Carga una configuración YAML y lanza un backtest rápido."""

    config = read_config(path)
    out = backtest_core(config)
    typer.secho(f"Backtest guardado en {out}", fg=typer.colors.GREEN)


@app.command()
def backtest(
    config: Optional[str] = typer.Option(None, help="Ruta YAML de configuración (opcional)."),
) -> None:
    """Ejecuta un backtest simple SMA cross sobre el primer ticker."""

    cfg = read_config(config)
    out = backtest_core(cfg)
    typer.secho(f"Backtest guardado en {out}", fg=typer.colors.GREEN)


@app.command()
def run_once(
    config: Optional[str] = typer.Option(None, help="Ruta YAML de configuración (opcional)."),
) -> None:
    """Evalúa el último cambio de señal y opera según la configuración."""

    cfg = read_config(config)
    run_once_core(cfg)


@app.command()
def plot(
    equity_csv: str = typer.Option(..., help="CSV con columnas: date,equity"),
    out: str = typer.Option("reports/equity_curve.png", help="Salida PNG"),
) -> None:
    """Genera la gráfica de la curva de capital."""

    plot_equity_curve(equity_csv, out)


@app.command()
def schedule(
    hour: int = typer.Option(22, help="Hora diaria (0-23)"),
    minute: int = typer.Option(0, help="Minuto (0-59)"),
    config: Optional[str] = typer.Option(None, help="Ruta YAML de configuración (opcional)."),
) -> None:
    """Programa la ejecución diaria de `run-once` usando APScheduler."""

    BlockingScheduler = _safe_import_apscheduler()
    if not BlockingScheduler:
        typer.secho("APScheduler no está instalado. Ejecuta: pip install apscheduler", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    cfg = read_config(config)

    def job() -> None:
        try:
            typer.secho(f"Ejecutando run-once @ {datetime.now().isoformat()}", fg=typer.colors.CYAN)
            run_once_core(cfg)
        except Exception as exc:
            typer.secho(f"Error en tarea programada: {exc}", fg=typer.colors.RED)

    scheduler = BlockingScheduler()
    scheduler.add_job(job, "cron", hour=hour, minute=minute)
    typer.secho(
        f"Planificador activo: {hour:02d}:{minute:02d} (zona del sistema). Ctrl+C para salir.",
        fg=typer.colors.BLUE,
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):  # pragma: no cover - interacción usuario
        typer.secho("Scheduler detenido.", fg=typer.colors.YELLOW)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    app()
