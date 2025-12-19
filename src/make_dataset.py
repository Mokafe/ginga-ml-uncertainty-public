#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_dataset.py
『銀河鉄道の夜』テキストを入力に、v2のCSV/JSONLを生成します。

ねらい（雑）：
- 先生の「凸レンズ模型」っぽく、中心ほど“厚い” -> 星が増える
- アルビレオの“重なり”っぽく、overlap が小さいほど観測が滲む（ノイズ↑）
- star_count はカウント -> Poisson（+一部だけ過分散を混ぜてもOK）
- haze_obs は連続 -> 回帰（+不確実性の話につなぐ）

※作品本文は同梱しません。各自で用意して --text に渡してください。
"""

import argparse, re, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ["銀河", "天の川", "星", "白", "青", "光", "望遠鏡", "レンズ", "サファイア", "トパース", "観測所"]

def split_sentences(text: str):
    parts = re.split(r"[。！？\n]+", text)
    return [p.strip() for p in parts if p.strip()]

def stable_hash01(s: str) -> float:
    h = int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)
    return (h % 10_000_000) / 10_000_000.0

def pick_pool(sents):
    pool = [s for s in sents if any(k in s for k in KEYS) and len(s) >= 18]
    return pool if len(pool) >= 50 else sents

def make_dataset(text: str, n: int, seed: int, add_event: bool):
    rng = np.random.default_rng(seed)
    sents = split_sentences(text)
    pool = pick_pool(sents)

    observers = ["ジョバンニ", "カムパネルラ", "先生"]
    instruments = ["両面凸レンズ模型", "アルビレオ両面凸レンズ"]

    # 先生の模型っぽい“厚み”：中心ほど厚い（ガウス）
    base, amp, sigma = 0.55, 1.25, 0.22

    rows = []
    for i in range(n):
        memo = pool[int(rng.integers(0, len(pool)))]
        observer = observers[int(rng.integers(0, len(observers)))]
        instrument = instruments[int(rng.integers(0, len(instruments)))]

        # 空の位置 u：本文メモに由来する揺らぎ＋少しノイズ
        u = (stable_hash01(memo) * 2 - 1) + rng.normal(0, 0.10)
        u = float(np.clip(u, -1.2, 1.2))

        # “重なり” overlap：回転の位相で 0..1
        t = int(rng.integers(0, 240))
        phase = 2*np.pi*(t/60.0)
        overlap = float((1 + np.cos(phase))/2)

        # 本文語のカウント
        f_white = memo.count("白") + memo.count("ぼんやり")
        f_blue  = memo.count("青") + memo.count("あお")
        f_light = memo.count("光") + memo.count("明")
        f_star  = memo.count("星")
        f_len   = len(memo)

        # 真の厚み（中心ほど厚い＋語感で微調整）
        thickness = base + amp*np.exp(-(u**2)/(2*sigma**2))
        thickness += 0.03*f_star + 0.015*f_light - 0.01*f_blue
        thickness = float(max(thickness, 0.05))

        # “青の深度”（厚み由来の潜在指標）
        depth_blue = float(1/(1+np.exp(-(thickness-1.0))))

        # 星数の期待値 λ（log-link）
        log_lambda = (-0.4
                      + 0.85*thickness
                      + 0.45*overlap
                      + 0.25*depth_blue
                      - 0.35*(u**2)
                      + 0.02*(f_len/50.0))
        true_lambda = float(np.exp(log_lambda))

        # 一部だけ過分散（議論のタネ）
        if rng.random() < 0.12:
            k = 6.0
            p = k/(k+true_lambda)
            star_count = int(rng.negative_binomial(k, p))
            overdisp = 1
        else:
            star_count = int(rng.poisson(true_lambda))
            overdisp = 0

        # ぼんやり白さ（連続）＋観測ノイズ：overlap小ほど滲む
        noise_sigma = float(0.08 + 0.22*(1-overlap) + 0.04*(1-depth_blue))
        haze_true = 0.25 + 0.55*thickness + 0.18*np.log1p(true_lambda) + 0.12*overlap + 0.06*f_white
        haze_obs  = float(haze_true + rng.normal(0, noise_sigma))

        rows.append({
            "obs_id": i,
            "observer": observer,
            "instrument": instrument,
            "t_index": t,
            "u": u,
            "overlap": overlap,
            "thickness_true": thickness,
            "depth_blue": depth_blue,
            "memo": memo,
            "f_white": f_white,
            "f_blue": f_blue,
            "f_light": f_light,
            "f_star": f_star,
            "f_len": f_len,
            "haze_obs": haze_obs,
            "star_count": star_count,
            "true_lambda": true_lambda,
            "noise_sigma": noise_sigma,
            "overdispersed": overdisp,
            "event_flag": 0,
            "event_name": "",
        })

    df = pd.DataFrame(rows)

    if add_event and len(df) > 0:
        # 強い外れ値1点（“観測の断絶”）
        i0 = int(df["thickness_true"].idxmax())
        anom = df.loc[i0].copy()
        anom["obs_id"] = int(df["obs_id"].max() + 1)
        anom["memo"] = "（観測の断絶）となりの席が急に空になったように、記録がうまく結ばれなかった。"
        anom["star_count"] = 0
        anom["haze_obs"] = float(df["haze_obs"].mean() + 4*df["haze_obs"].std())
        anom["event_flag"] = 1
        anom["event_name"] = "観測の断絶"
        df = pd.concat([df, anom.to_frame().T], ignore_index=True)

    return df

def write_jsonl(df: pd.DataFrame, path: Path):
    with path.open("w", encoding="utf-8") as f:
        for r in df.to_dict(orient="records"):
            obj = {
                "id": int(r["obs_id"]),
                "meta": {
                    "observer": r["observer"],
                    "instrument": r["instrument"],
                    "t_index": int(r["t_index"]),
                    "memo": r["memo"],
                },
                "inputs": {
                    "u": float(r["u"]),
                    "overlap": float(r["overlap"]),
                    "depth_blue": float(r["depth_blue"]),
                    "f_white": int(r["f_white"]),
                    "f_blue": int(r["f_blue"]),
                    "f_light": int(r["f_light"]),
                    "f_star": int(r["f_star"]),
                    "f_len": int(r["f_len"]),
                },
                "targets": {
                    "haze_obs": float(r["haze_obs"]),
                    "star_count": int(r["star_count"]),
                },
                "event": {
                    "flag": int(r.get("event_flag", 0)),
                    "name": r.get("event_name", ""),
                },
                "ground_truth": {
                    "thickness_true": float(r["thickness_true"]),
                    "true_lambda": float(r["true_lambda"]),
                    "noise_sigma": float(r["noise_sigma"]),
                    "overdispersed": int(r["overdispersed"]),
                }
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="入力テキスト（UTF-8）")
    ap.add_argument("--out_csv", default="ginga_galaxy_haze_v2.csv")
    ap.add_argument("--out_jsonl", default="ginga_galaxy_haze_v2.jsonl")
    ap.add_argument("--n", type=int, default=1400)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--add_event", action="store_true", help="外れ値1点を混ぜる")
    args = ap.parse_args()

    text = Path(args.text).read_text(encoding="utf-8")
    df = make_dataset(text=text, n=args.n, seed=args.seed, add_event=args.add_event)

    out_csv = Path(args.out_csv)
    out_jsonl = Path(args.out_jsonl)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_csv, index=False, encoding="utf-8")
    write_jsonl(df, out_jsonl)

    print("wrote:", out_csv)
    print("wrote:", out_jsonl)

if __name__ == "__main__":
    main()
