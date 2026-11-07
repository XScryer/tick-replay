"""Download historical bid/ask ticks via the cTrader Open API (Spotware).

Why this exists: most brokers/platforms only serve aggregated bars or
simulated ticks for free. The cTrader Open API serves the *real* tick feed
(bid and ask separately) going back years — on a demo account — which makes
it possible to backtest on the same quotes a live bot would see.

Protocol notes (verified empirically and per official docs):
  ProtoOAGetTickDataReq: symbolId + type (BID/ASK) + from/to timestamps (ms).
  The tickData list comes NEWEST-FIRST; the first timestamp is an absolute
  epoch-ms and the following ones are DELTAS in ms (time decreases).
  Price comes as an integer `tick`: price = tick / 10^digits.
  hasMore=true -> older ticks exist: repeat with
  toTimestamp = oldest_received - 1.
  Official rate limit: 5 historical requests/second per connection.
  A single request cannot span more than one week — this downloader chunks
  by day, which also gives natural resume checkpoints.

Usage:
    # 1. create an app at https://openapi.ctrader.com/apps (free, demo OK)
    # 2. put credentials in .env (see .env.example)
    # 3. first run opens a browser for OAuth, tokens are cached in .env
    python -m tickreplay.ctrader --symbol EURUSD --from 2020-01-01 --to 2024-12-31 --out data/EURUSD/raw
    python -m tickreplay.data csv2bin --raw-dir data/EURUSD/raw --symbol EURUSD --bin-dir data/EURUSD/bin
"""

import argparse
import calendar
import datetime as dt
import os
import sys
import time

ENV_PATH = ".env"
REQUIRED_ENV = ("OPEN_API_CLIENT_ID", "OPEN_API_SECRET")
TOKEN_KEYS = ("CTRADER_ACCESS_TOKEN", "CTRADER_REFRESH_TOKEN")
REDIRECT_URI = "http://localhost:8899/callback"
CALLBACK_PORT = 8899


def load_env(path=ENV_PATH):
    """Read KEY=VALUE lines; real environment variables win over the file."""
    out = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    out[k.strip()] = v.strip()
    out.update({k: v for k, v in os.environ.items() if k in REQUIRED_ENV + TOKEN_KEYS})
    return out


def save_env_tokens(tokens, path=ENV_PATH):
    """Persist OAuth tokens back into .env (creating it if needed)."""
    lines = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]
    merged = dict(tokens)
    seen = set()
    new_lines = []
    for ln in lines:
        if "=" in ln and not ln.strip().startswith("#"):
            k = ln.split("=", 1)[0].strip()
            if k in merged:
                new_lines.append(f"{k}={merged.pop(k)}")
                seen.add(k)
                continue
        new_lines.append(ln)
    for k in TOKEN_KEYS:
        if k in merged:
            new_lines.append(f"{k}={merged[k]}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")


def oauth_flow(env):
    """Interactive first-time auth: opens browser, catches ?code= on localhost.

    Returns {"access": ..., "refresh": ...}.
    """
    from ctrader_open_api import Auth
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from threading import Thread
    from urllib.parse import urlparse, parse_qs

    auth = Auth(env["OPEN_API_CLIENT_ID"], env["OPEN_API_SECRET"], REDIRECT_URI)
    result = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            qs = parse_qs(urlparse(self.path).query)
            if "code" in qs:
                result["code"] = qs["code"][0]
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"OK - you can close this tab and return to the terminal.")
            else:
                self.send_response(400)
                self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("localhost", CALLBACK_PORT), Handler)
    Thread(target=srv.serve_forever, daemon=True).start()
    url = auth.getAuthUri()
    print("OPEN THIS URL IN YOUR BROWSER AND APPROVE:", url)
    import webbrowser
    webbrowser.open(url)
    t0 = time.time()
    while "code" not in result and time.time() - t0 < 600:
        time.sleep(0.5)
    srv.shutdown()
    if "code" not in result:
        raise RuntimeError("OAuth timed out after 10 minutes")
    token_resp = auth.getToken(result["code"])
    if "accessToken" not in token_resp:
        raise RuntimeError(f"code exchange failed: {token_resp}")
    return {"access": token_resp["accessToken"], "refresh": token_resp.get("refreshToken", "")}


def _ts_ms(d):
    return int(calendar.timegm(d.utctimetuple())) * 1000


def download_ticks(symbol, date_from, date_to, out_dir,
                   probe=False, max_requests=100000, env_path=ENV_PATH):
    """Download BID+ASK ticks for `symbol` between the two dates (inclusive).

    Writes <symbol>_ticks_<from>_<to>_<SIDE>.csv (columns time_ms,time,price)
    plus a _progress.txt checkpoint so interrupted runs resume where they
    stopped. Set probe=True to count/sample without writing anything.
    """
    env = load_env(env_path)
    missing = [k for k in REQUIRED_ENV if not env.get(k)]
    if missing:
        raise RuntimeError(
            f"missing {missing} — copy .env.example to .env and fill in your "
            f"cTrader Open API app credentials")

    tokens = {k: env[k] for k in TOKEN_KEYS if env.get(k)}
    if not tokens.get("CTRADER_ACCESS_TOKEN"):
        token_resp = oauth_flow(env)
        tokens.update({"CTRADER_ACCESS_TOKEN": token_resp["access"],
                       "CTRADER_REFRESH_TOKEN": token_resp["refresh"]})
        save_env_tokens(tokens, env_path)
    access = tokens["CTRADER_ACCESS_TOKEN"]

    from ctrader_open_api import Client, TcpProtocol, EndPoints, Protobuf
    from ctrader_open_api.messages import OpenApiMessages_pb2 as OA
    from ctrader_open_api.messages import OpenApiModelMessages_pb2 as OAM
    from ctrader_open_api.messages import OpenApiCommonMessages_pb2 as OAC
    from twisted.internet import reactor

    d_start = dt.datetime.strptime(date_from, "%Y-%m-%d")
    end_dt = dt.datetime.strptime(date_to, "%Y-%m-%d") + dt.timedelta(days=1)

    ctx = {
        "account_id": "", "symbolId": None, "digits": 5,
        "error": None, "done": False, "requests": 0,
        "queue": [], "current": None, "buffer": {},
    }

    client = Client(EndPoints.PROTOBUF_LIVE_HOST, EndPoints.PROTOBUF_PORT,
                    TcpProtocol)

    def heartbeat():
        if not ctx.get("done"):
            try:
                client.send(OAC.ProtoHeartbeatEvent(),
                            responseTimeoutInSeconds=60).addErrback(lambda f: None)
            except Exception:
                pass
            reactor.callLater(10, heartbeat)

    def fail(f, stage):
        print(f"[ERROR {stage}]", repr(f))
        ctx["error"] = stage
        shutdown()

    def on_api_error(msg, stage):
        print(f"[ERROR {stage}]", Protobuf.extract(msg))
        ctx["error"] = stage
        shutdown()

    # ---- flow: app auth -> account list -> account auth -> symbols -> ticks
    def step_app_auth(_):
        req = OA.ProtoOAGetAccountListByAccessTokenReq()
        req.accessToken = access
        client.send(req, responseTimeoutInSeconds=20).addCallbacks(
            step_account_list, lambda f: fail(f, "account_list"))

    def step_account_list(msg):
        if msg.payloadType == OA.ProtoOAErrorRes().payloadType:
            return on_api_error(msg, "account_list")
        res = Protobuf.extract(msg)
        accounts = list(getattr(res, "ctidTraderAccount", []))
        live = [c for c in accounts if getattr(c, "isLive", False)]
        chosen = live[0] if live else accounts[0]
        ctx["account_id"] = str(chosen.ctidTraderAccountId)
        print(f"account: id={ctx['account_id']}")
        req = OA.ProtoOAAccountAuthReq()
        req.ctidTraderAccountId = int(ctx["account_id"])
        req.accessToken = access
        client.send(req, responseTimeoutInSeconds=20).addCallbacks(
            step_symbols, lambda f: fail(f, "account_auth"))

    def step_symbols(_):
        req = OA.ProtoOASymbolsListReq()
        req.ctidTraderAccountId = int(ctx["account_id"])
        req.includeArchivedSymbols = False
        client.send(req, responseTimeoutInSeconds=30).addCallbacks(
            step_pick_symbol, lambda f: fail(f, "symbols"))

    def step_pick_symbol(msg):
        if msg.payloadType == OA.ProtoOAErrorRes().payloadType:
            return on_api_error(msg, "symbols")
        res = Protobuf.extract(msg)
        found = None
        for s in list(getattr(res, "symbol", [])) + \
                 list(getattr(res, "archivedSymbol", [])):
            if getattr(s, "symbolName", "") == symbol:
                found = s
                break
        if found is None:
            print("Symbol not found:", symbol)
            shutdown()
            return
        ctx["symbolId"] = found.symbolId
        ctx["digits"] = getattr(found, "digits", 5)
        print(f"{symbol}: symbolId={ctx['symbolId']} digits={ctx['digits']}")

        # Empirical discovery (probe 2025-06-02): chunkSize=10000;
        # tickData NEWEST-FIRST; 1st timestamp=absolute ms, following ones
        # are negative ms deltas; 1st tick=absolute price x10^digits, the
        # following ones are price deltas. One EURUSD day ~= 100-250k
        # ticks/side -> queue of 1 item per day per side, flushed on
        # completion (bounded memory).
        days = []
        d = d_start
        while d < end_dt:
            if d.weekday() != 5:  # Saturday: forex closed, don't waste calls
                next_d = min(d + dt.timedelta(days=1), end_dt)
                days.append((_ts_ms(d), _ts_ms(next_d) - 1))  # [d, next_d)
            d += dt.timedelta(days=1)

        # resume: _progress.txt lists "YYYY-MM-DD SIDE" already written
        prog_path = os.path.join(out_dir, "_progress.txt")
        done_set = set()
        if os.path.exists(prog_path):
            with open(prog_path, "r", encoding="utf-8") as f:
                for ln in f:
                    p = ln.strip().split()
                    if len(p) == 2:
                        done_set.add((p[0], p[1]))
        ctx["prog_path"] = prog_path
        ctx["queue"] = []
        for a, b in days:
            day_str = dt.datetime.fromtimestamp(a / 1000, dt.timezone.utc).strftime("%Y-%m-%d")
            for t in ("BID", "ASK"):
                if (day_str, t) not in done_set:
                    ctx["queue"].append((a, b, t))
        ctx["skipped"] = len(days) * 2 - len(ctx["queue"])
        print(f"items to download: {len(ctx['queue'])} "
              f"({ctx['skipped']} already done)")
        ctx["current"] = None
        ctx["buffer"] = {}          # ts_ms -> price_int for current item
        ctx["item_side"] = None
        ctx["file_bid"] = os.path.join(
            out_dir, f"{symbol}_ticks_{date_from}_{date_to}_BID.csv")
        ctx["file_ask"] = os.path.join(
            out_dir, f"{symbol}_ticks_{date_from}_{date_to}_ASK.csv")
        ctx["written"] = {"BID": 0, "ASK": 0}
        if not probe:
            os.makedirs(out_dir, exist_ok=True)
            for k in ("bid", "ask"):
                path = ctx[f"file_{k}"]
                # APPEND if it exists: opening with "w" truncated everything
                # on every resume while _progress.txt kept days marked =
                # SILENT DATA LOSS
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    continue
                with open(path, "w", encoding="utf-8", newline="") as f:
                    f.write("time_ms,time,price\n")
        next_page()

    # ---- tick pagination ----
    # Official docs (help.ctrader.com/open-api/messages):
    #   tickData arrives NEWEST-FIRST. The 1st timestamp is absolute ms; the
    #   following ones are ms DELTAS vs the previous tick (time decreases).
    #   Rebuild: t[0]=abs; t[i]=t[i-1]-delta[i].
    #   hasMore=true -> the response filled chunkSize and older ticks exist:
    #   request again with toTimestamp = oldest_received - 1.
    #   Official rate limit: 5 req/s for historical requests.
    def next_page():
        if ctx.get("done"):
            return
        if ctx["requests"] >= max_requests:
            print("request limit reached")
            finish_and_report()
            return
        if ctx["current"] is None:
            if not ctx["queue"]:
                finish_and_report()
                return
            ctx["current"] = ctx["queue"].pop(0)
            ctx["item_side"] = ctx["current"][2]
            ctx["buffer"] = {}
        win_start, win_end, side = ctx["current"]
        if win_start >= win_end:
            flush_item()
            ctx["current"] = None
            next_page()
            return
        req = OA.ProtoOAGetTickDataReq()
        req.ctidTraderAccountId = int(ctx["account_id"])
        req.symbolId = ctx["symbolId"]
        req.type = getattr(OAM.ProtoOAQuoteType, side)
        req.fromTimestamp = win_start
        req.toTimestamp = win_end
        ctx["requests"] += 1
        ctx["req_from_ms"], ctx["req_to_ms"], ctx["req_side"] = win_start, win_end, side
        if ctx["requests"] <= 3:
            d1 = dt.datetime.fromtimestamp(win_start / 1000, dt.timezone.utc)
            d2 = dt.datetime.fromtimestamp(win_end / 1000, dt.timezone.utc)
            print(f"  req {side}: {d1} -> {d2}")
        client.send(req, responseTimeoutInSeconds=60).addCallbacks(
            step_ticks, lambda f: fail(f, "tickdata"))

    def step_ticks(msg):
        if msg.payloadType == OA.ProtoOAErrorRes().payloadType:
            err_info = Protobuf.extract(msg)
            d1 = dt.datetime.fromtimestamp(ctx["req_from_ms"] / 1000, dt.timezone.utc)
            d2 = dt.datetime.fromtimestamp(ctx["req_to_ms"] / 1000, dt.timezone.utc)
            print(f"[ERROR tickdata {ctx.get('req_side')} {d1} -> {d2}]", err_info)
            if "LIMIT" in str(err_info).upper() or "429" in str(err_info):
                print("  rate limit: waiting 10s and retrying")
                reactor.callLater(10, next_page)
                return
            # other error: retry the SAME request; only give up on the whole
            # day after 5 failures — never write a partial day or mark
            # progress
            ctx["item_errors"] = ctx.get("item_errors", 0) + 1
            if ctx["item_errors"] <= 5:
                print(f"  non-limit error: retry {ctx['item_errors']}/5 in 10s")
                reactor.callLater(10, next_page)
                return
            ctx.setdefault("gaps", []).append(
                (ctx["req_from_ms"], ctx["req_to_ms"], ctx["req_side"]))
            print(f"  [GAP] day discarded without writing: "
                  f"{ctx.get('req_side')} "
                  f"{dt.datetime.fromtimestamp(ctx['req_from_ms'] / 1000, dt.timezone.utc)} -> "
                  f"{dt.datetime.fromtimestamp(ctx['req_to_ms'] / 1000, dt.timezone.utc)}")
            ctx["buffer"] = {}
            ctx["current"] = None
            ctx["item_errors"] = 0
            next_page()
            return
        res = Protobuf.extract(msg)
        ctx["item_errors"] = 0
        tick_list = list(getattr(res, "tickData", []))
        side = ctx["req_side"]
        hasMore = getattr(res, "hasMore", False)
        if tick_list:
            # rebuild absolute times and absolute prices (newest-first)
            times = [tick_list[0].timestamp]
            prices = [tick_list[0].tick]
            for i in range(1, len(tick_list)):
                times.append(times[-1] + tick_list[i].timestamp)  # ms delta (<0)
                prices.append(prices[-1] + tick_list[i].tick)       # price delta
            if ctx["requests"] <= 3:
                print(f"    chunk {len(tick_list)} ticks, "
                      f"newest={dt.datetime.fromtimestamp(times[0] / 1000, dt.timezone.utc)}, "
                      f"oldest={dt.datetime.fromtimestamp(times[-1] / 1000, dt.timezone.utc)}, "
                      f"hasMore={hasMore}")
                for tp, pr in zip(times[:3], prices[:3]):
                    print(f"      ts={tp} price_int={pr} -> "
                          f"{pr / (10 ** ctx['digits']):.{ctx['digits']}f}")
            for tp_abs, pr in zip(times, prices):
                ctx["buffer"][tp_abs] = pr
            if not hasMore:
                # hasMore=false -> ALL ticks of the interval: item complete
                flush_item()
                ctx["current"] = None
            else:
                # next page: same from, to = oldest - 1
                ctx["current"] = (ctx["current"][0], times[-1] - 1, side)
                if ctx["current"][0] >= ctx["current"][1]:
                    flush_item()
                    ctx["current"] = None
        else:
            # empty chunk: step toTimestamp back 1 day to avoid getting stuck
            win0 = ctx["current"][0]
            new_to = ctx["current"][1] - 86400000
            if new_to <= win0:
                flush_item()
                ctx["current"] = None
            else:
                ctx["current"] = (win0, new_to, side)
        if ctx["requests"] % 20 == 0:
            print(f"  {ctx['requests']} requests, "
                  f"{sum(ctx['written'].values())} ticks written, "
                  f"side={side}, hasMore={hasMore}")
        next_page()

    def flush_item():
        """Write the current item's buffer (1 day x side) to CSV and clear."""
        if probe or ctx["current"] is None:
            ctx["buffer"] = {}
            return
        side = ctx["item_side"]
        csv_path = ctx["file_bid"] if side == "BID" else ctx["file_ask"]
        scale = 10 ** ctx["digits"]
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            for ts_ in sorted(ctx["buffer"]):
                d = dt.datetime.fromtimestamp(ts_ / 1000, dt.timezone.utc)
                f.write(f"{ts_},{d.strftime('%Y.%m.%d %H:%M:%S.%f')[:-3]},"
                        f"{ctx['buffer'][ts_] / scale:.{ctx['digits']}f}\n")
        ctx["written"][side] += len(ctx["buffer"])
        # checkpoint: mark the item as done for resume
        day_str = dt.datetime.fromtimestamp(
            ctx["current"][0] / 1000, dt.timezone.utc).strftime("%Y-%m-%d")
        with open(ctx["prog_path"], "a", encoding="utf-8") as f:
            f.write(f"{day_str} {side}\n")
        ctx["buffer"] = {}

    def finish_and_report():
        n = sum(ctx["written"].values())
        print(f"DONE: {ctx['requests']} requests, {n} ticks "
              f"(BID={ctx['written']['BID']}, ASK={ctx['written']['ASK']}), "
              f"queue left={len(ctx['queue'])}, gaps={len(ctx.get('gaps', []))}")
        if n == 0:
            print("RESULT: no ticks returned by the API for this period.")
        else:
            if ctx.get("gaps"):
                print(f"  gaps ({len(ctx['gaps'])}): {ctx['gaps'][:5]}")
            print(f"  BID -> {ctx['file_bid']}")
            print(f"  ASK -> {ctx['file_ask']}")
        shutdown()

    def shutdown():
        ctx["done"] = True
        try:
            reactor.stop()
        except Exception:
            pass

    req = OA.ProtoOAApplicationAuthReq()
    req.clientId = env["OPEN_API_CLIENT_ID"]
    req.clientSecret = env["OPEN_API_SECRET"]
    client.send(req, responseTimeoutInSeconds=30).addCallbacks(
        lambda m: (on_api_error(m, "app_auth")
                   if m.payloadType == OA.ProtoOAErrorRes().payloadType
                   else step_app_auth(m)),
        lambda f: fail(f, "app_auth"))
    client.startService()
    client.setConnectedCallback(lambda c: print("connected"))
    client.setDisconnectedCallback(lambda c, r: print("disconnected:", r))
    heartbeat()
    reactor.run()
    if ctx.get("error"):
        raise RuntimeError(f"download failed at stage: {ctx['error']}")


def _cli():
    ap = argparse.ArgumentParser(
        prog="tickreplay.ctrader",
        description="Download historical bid/ask ticks via cTrader Open API")
    ap.add_argument("--symbol", required=True, help="e.g. EURUSD")
    ap.add_argument("--from", dest="date_from", required=True,
                    help="start date YYYY-MM-DD (UTC)")
    ap.add_argument("--to", dest="date_to", required=True,
                    help="end date YYYY-MM-DD (UTC, inclusive)")
    ap.add_argument("--out", required=True,
                    help="output dir for raw CSVs (use one dir per symbol)")
    ap.add_argument("--probe", action="store_true",
                    help="count/sample requests without writing files")
    ap.add_argument("--max-requests", type=int, default=100000)
    ap.add_argument("--env", default=ENV_PATH, help="path to .env file")
    args = ap.parse_args()
    download_ticks(args.symbol, args.date_from, args.date_to, args.out,
                   probe=args.probe, max_requests=args.max_requests,
                   env_path=args.env)


if __name__ == "__main__":
    _cli()
