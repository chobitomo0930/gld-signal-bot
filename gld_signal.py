#!/usr/bin/env python3
"""
GLD スイングトレード シグナル検知 & LINE通知
GitHub Actions + yfinance + LINE Messaging API

戦略（スコアリング方式 — 5条件中2つ以上で買い）:
  条件1: GLD RSI(14) <= 50
  条件2: SOXL 3日間変動率 <= -3%
  条件3: VIX > VIX MA20（恐怖上昇中）
  条件4: GLD価格 <= BB下限 x 1.02
  条件5: GLD MA20 > MA50（短期上昇トレンド）

  利確: RSI >= 75 or +5%
  損切り: 初期-7%, トレーリングストップ4%（高値から4%下落）

状態管理:
  state.json にポジション情報を保持し、GitHub Actions上でgit commitで永続化
"""

import os
import json
import subprocess
import urllib.request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ============================================
# 環境変数から設定を読み込み（GitHub Secrets）
# ============================================
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_USER_ID = os.environ.get("LINE_USER_ID", "")

# ============================================
# トレード設定
# ============================================
SYMBOL = "GLD"

RSI_PERIOD = 14
RSI_BUY_THRESH = 50
RSI_SELL = 75
SCORE_THRESHOLD = 2

SOXL_DROP_DAYS = 3
SOXL_DROP_PCT = 3

BB_PERIOD = 20
BB_NUM_STD = 2
BB_MARGIN = 1.02

MA_SHORT = 20
MA_LONG = 50

VIX_MA_PERIOD = 20

TAKE_PROFIT_PCT = 0.05
STOP_LOSS_PCT = 0.07
TRAILING_STOP_PCT = 0.04

STATE_FILE = Path(__file__).parent / "state.json"

# ============================================
# 状態管理（JSON永続化）
# ============================================
def load_state():
    default = {
        "position": False,
        "entry_price": 0.0,
        "entry_date": "",
        "peak_price": 0.0,
        "stop_loss_price": 0.0,
        "shares": 0,
    }
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
            for k, v in default.items():
                if k not in state:
                    state[k] = v
            return state
        except (json.JSONDecodeError, IOError):
            return default
    return default


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    print(f"[STATE] 保存完了: {STATE_FILE}")


def commit_state():
    if not os.environ.get("GITHUB_ACTIONS"):
        print("[STATE] ローカル実行のためgit pushスキップ")
        return

    try:
        subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=True)
        subprocess.run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"], check=True)
        subprocess.run(["git", "add", str(STATE_FILE)], check=True)

        result = subprocess.run(["git", "diff", "--cached", "--quiet"], capture_output=True)
        if result.returncode != 0:
            subprocess.run(["git", "commit", "-m", "Update state.json [skip ci]"], check=True)
            subprocess.run(["git", "push"], check=True)
            print("[STATE] git push 完了")
        else:
            print("[STATE] 変更なし、pushスキップ")
    except subprocess.CalledProcessError as e:
        print(f"[STATE] git操作失敗: {e}")


# ============================================
# テクニカル指標計算
# ============================================
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_bollinger(series, period=20, num_std=2):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return ma, upper, lower


# ============================================
# LINE Messaging API
# ============================================
def send_line_message(message_text):
    if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_USER_ID:
        print("[ERROR] LINE credentials not set.")
        return False

    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"
    }
    body = {
        "to": LINE_USER_ID,
        "messages": [{"type": "text", "text": message_text}]
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as res:
            print(f"[LINE] 送信成功 (status: {res.status})")
            return True
    except urllib.error.HTTPError as e:
        print(f"[LINE] 送信失敗: {e.code} {e.read().decode()}")
        return False


# ============================================
# メイン：シグナル判定
# ============================================
def check_signal():
    jst = timezone(timedelta(hours=9))
    now_jst = datetime.now(jst).strftime("%Y-%m-%d %H:%M JST")

    print(f"=== GLD Signal Check [{now_jst}] ===")

    # --- 状態読み込み ---
    state = load_state()
    print(f"  Position: {'HOLDING' if state['position'] else 'NONE'}")
    if state["position"]:
        print(f"    Entry: ${state['entry_price']} ({state['entry_date']})")
        print(f"    Peak: ${state['peak_price']}")
        print(f"    Stop Loss: ${state['stop_loss_price']}")

    # --- GLD 日足データ取得 ---
    try:
        gld = yf.Ticker(SYMBOL)
        df_gld = gld.history(period="1y", interval="1d")
    except Exception as e:
        print(f"[ERROR] GLD 日足データ取得失敗: {e}")
        return

    if df_gld.empty or len(df_gld) < MA_LONG + 1:
        print("[ERROR] GLD 日足データが不十分です")
        return

    if df_gld.index.tz is not None:
        df_gld.index = df_gld.index.tz_localize(None)

    gld_close = df_gld["Close"]
    gld_rsi = calc_rsi(gld_close, RSI_PERIOD)
    gld_ma20 = gld_close.rolling(MA_SHORT).mean()
    gld_ma50 = gld_close.rolling(MA_LONG).mean()
    _, _, bb_lower = calc_bollinger(gld_close, BB_PERIOD, BB_NUM_STD)

    # --- SOXL データ取得（逆相関シグナル用） ---
    try:
        soxl = yf.Ticker("SOXL")
        df_soxl = soxl.history(period="1y", interval="1d")
    except Exception as e:
        print(f"[ERROR] SOXL データ取得失敗: {e}")
        return

    if df_soxl.empty or len(df_soxl) < SOXL_DROP_DAYS + 1:
        print("[ERROR] SOXL データが不十分です")
        return

    if df_soxl.index.tz is not None:
        df_soxl.index = df_soxl.index.tz_localize(None)

    soxl_close = df_soxl["Close"].reindex(df_gld.index, method="ffill")
    soxl_change_3d = soxl_close.pct_change(SOXL_DROP_DAYS) * 100

    # --- VIX データ取得 ---
    try:
        vix = yf.Ticker("^VIX")
        df_vix = vix.history(period="1y", interval="1d")
    except Exception as e:
        print(f"[ERROR] VIX データ取得失敗: {e}")
        return

    if df_vix.empty:
        print("[ERROR] VIX データが不十分です")
        return

    if df_vix.index.tz is not None:
        df_vix.index = df_vix.index.tz_localize(None)

    vix_close = df_vix["Close"].reindex(df_gld.index, method="ffill")
    vix_ma20 = vix_close.rolling(VIX_MA_PERIOD).mean()

    # --- 各指標の最新値 ---
    price = round(gld_close.iloc[-1], 2)
    rsi_val = round(gld_rsi.iloc[-1], 2)
    ma20_val = round(gld_ma20.iloc[-1], 2)
    ma50_val = round(gld_ma50.iloc[-1], 2)
    bb_lower_val = round(bb_lower.iloc[-1], 2)
    soxl_chg = round(soxl_change_3d.iloc[-1], 1) if not pd.isna(soxl_change_3d.iloc[-1]) else None
    vix_val = round(vix_close.iloc[-1], 2) if not pd.isna(vix_close.iloc[-1]) else None
    vix_ma_val = round(vix_ma20.iloc[-1], 2) if not pd.isna(vix_ma20.iloc[-1]) else None

    # --- スコアリング ---
    score = 0
    score_details = []

    cond1 = rsi_val <= RSI_BUY_THRESH
    if cond1:
        score += 1
    score_details.append(f"RSI({rsi_val}) <= {RSI_BUY_THRESH}: {'YES' if cond1 else 'No'}")

    cond2 = soxl_chg is not None and soxl_chg <= -SOXL_DROP_PCT
    if cond2:
        score += 1
    score_details.append(f"SOXL 3d({soxl_chg}%) <= -{SOXL_DROP_PCT}%: {'YES' if cond2 else 'No'}")

    cond3 = vix_val is not None and vix_ma_val is not None and vix_val > vix_ma_val
    if cond3:
        score += 1
    score_details.append(f"VIX({vix_val}) > MA20({vix_ma_val}): {'YES' if cond3 else 'No'}")

    cond4 = price <= bb_lower_val * BB_MARGIN
    if cond4:
        score += 1
    score_details.append(f"Price({price}) <= BB_Low*1.02({round(bb_lower_val * BB_MARGIN, 2)}): {'YES' if cond4 else 'No'}")

    cond5 = ma20_val > ma50_val
    if cond5:
        score += 1
    score_details.append(f"MA20({ma20_val}) > MA50({ma50_val}): {'YES' if cond5 else 'No'}")

    # --- コンソール出力 ---
    print(f"  Price:  ${price}")
    print(f"  RSI(14):   {rsi_val}")
    print(f"  MA20: ${ma20_val}  MA50: ${ma50_val}")
    print(f"  BB Lower: ${bb_lower_val}")
    print(f"  SOXL 3d Change: {soxl_chg}%")
    print(f"  VIX: {vix_val} (MA20: {vix_ma_val})")
    print(f"  --- Score: {score}/{SCORE_THRESHOLD} ---")
    for d in score_details:
        print(f"    {d}")

    # --- ポジション保有中の処理 ---
    state_changed = False

    if state["position"]:
        if price > state["peak_price"]:
            state["peak_price"] = price
            new_stop = round(price * (1 - TRAILING_STOP_PCT), 2)
            if new_stop > state["stop_loss_price"]:
                state["stop_loss_price"] = new_stop
                state_changed = True
                print(f"  Trailing updated: peak ${price} -> SL ${new_stop}")

        entry_price = state["entry_price"]
        pnl_pct = round((price / entry_price - 1) * 100, 1) if entry_price > 0 else 0

        # 損切り判定（トレーリングストップ or 初期ストップ）
        if price <= state["stop_loss_price"]:
            msg = (
                f"🛑 GLD 損切りライン到達！\n"
                f"━━━━━━━━━━━━━━\n"
                f"📊 現在値: ${price}\n"
                f"📍 エントリー: ${entry_price} ({state['entry_date']})\n"
                f"📉 損益: {pnl_pct:+.1f}%\n"
                f"🛑 損切りライン: ${state['stop_loss_price']}\n"
                f"━━━━━━━━━━━━━━\n"
                f"💰 ポジションの決済を検討\n"
                f"⏰ {now_jst}"
            )
            send_line_message(msg)

            state = {"position": False, "entry_price": 0, "entry_date": "",
                     "peak_price": 0, "stop_loss_price": 0, "shares": 0}
            state_changed = True

        # RSI利確判定
        elif rsi_val >= RSI_SELL:
            msg = (
                f"🟡 GLD RSI利確シグナル発生！\n"
                f"━━━━━━━━━━━━━━\n"
                f"📊 現在値: ${price}\n"
                f"📍 エントリー: ${entry_price} ({state['entry_date']})\n"
                f"📈 損益: {pnl_pct:+.1f}%\n"
                f"📉 RSI(14): {rsi_val} (>={RSI_SELL})\n"
                f"━━━━━━━━━━━━━━\n"
                f"💰 保有ポジションの利確を検討\n"
                f"⏰ {now_jst}"
            )
            send_line_message(msg)

            state = {"position": False, "entry_price": 0, "entry_date": "",
                     "peak_price": 0, "stop_loss_price": 0, "shares": 0}
            state_changed = True

        # +5%利確判定
        elif pnl_pct >= TAKE_PROFIT_PCT * 100:
            msg = (
                f"🟡 GLD 利確ライン到達！\n"
                f"━━━━━━━━━━━━━━\n"
                f"📊 現在値: ${price}\n"
                f"📍 エントリー: ${entry_price} ({state['entry_date']})\n"
                f"📈 損益: {pnl_pct:+.1f}% (>= +{int(TAKE_PROFIT_PCT*100)}%)\n"
                f"━━━━━━━━━━━━━━\n"
                f"💰 保有ポジションの利確を検討\n"
                f"⏰ {now_jst}"
            )
            send_line_message(msg)

            state = {"position": False, "entry_price": 0, "entry_date": "",
                     "peak_price": 0, "stop_loss_price": 0, "shares": 0}
            state_changed = True

        else:
            print(f"  Holding: P&L {pnl_pct:+.1f}% (SL: ${state['stop_loss_price']})")
            print("  -> No signal. LINE skip.")

    # --- ポジションなし → 買いシグナル判定 ---
    else:
        buy_signal = score >= SCORE_THRESHOLD

        print(f"  Buy Signal: {'YES' if buy_signal else 'No'} (Score {score}/{SCORE_THRESHOLD})")

        if buy_signal:
            stop_loss = round(price * (1 - STOP_LOSS_PCT), 2)

            # 該当条件をLINE通知に表示
            active_conditions = []
            if cond1:
                active_conditions.append(f"RSI({rsi_val}) <= {RSI_BUY_THRESH}")
            if cond2:
                active_conditions.append(f"SOXL 3日間{soxl_chg:+.1f}%")
            if cond3:
                active_conditions.append(f"VIX({vix_val}) > MA20({vix_ma_val})")
            if cond4:
                active_conditions.append(f"BB下限タッチ")
            if cond5:
                active_conditions.append(f"MA20 > MA50 (上昇トレンド)")
            conditions_str = "\n".join(f"  ✅ {c}" for c in active_conditions)

            msg = (
                f"🟢 GLD 買いシグナル発生！\n"
                f"━━━━━━━━━━━━━━\n"
                f"📊 現在値: ${price}\n"
                f"🎯 スコア: {score}/5 (閾値: {SCORE_THRESHOLD})\n"
                f"━━━━━━━━━━━━━━\n"
                f"📋 該当条件:\n"
                f"{conditions_str}\n"
                f"━━━━━━━━━━━━━━\n"
                f"🛑 初期損切り: ${stop_loss} (-{int(STOP_LOSS_PCT*100)}%)\n"
                f"📐 トレーリング: 高値から-{int(TRAILING_STOP_PCT*100)}%\n"
                f"🎯 利確目標: +{int(TAKE_PROFIT_PCT*100)}% or RSI>={RSI_SELL}\n"
                f"━━━━━━━━━━━━━━\n"
                f"💰 SBI証券で買い注文を検討\n"
                f"⏰ {now_jst}"
            )
            send_line_message(msg)

            entry_date = datetime.now(jst).strftime("%Y-%m-%d")
            state = {
                "position": True,
                "entry_price": price,
                "entry_date": entry_date,
                "peak_price": price,
                "stop_loss_price": stop_loss,
                "shares": 0,
            }
            state_changed = True

        else:
            print("  -> No signal. LINE skip.")

    # --- 状態保存 ---
    if state_changed:
        save_state(state)
        commit_state()

    print("=== Check Complete ===")


# ============================================
# 実行
# ============================================
if __name__ == "__main__":
    check_signal()
