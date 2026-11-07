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
    tok = auth.getToken(result["code"])
    if "accessToken" not in tok:
        raise RuntimeError(f"code exchange failed: {tok}")
    return {"access": tok["accessToken"], "refresh": tok.get("refreshToken", "")}


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
        tok = oauth_flow(env)
        tokens.update({"CTRADER_ACCESS_TOKEN": tok["access"],
                       "CTRADER_REFRESH_TOKEN": tok["refresh"]})
        save_env_tokens(tokens, env_path)
    access = tokens["CTRADER_ACCESS_TOKEN"]

    from ctrader_open_api import Client, TcpProtocol, EndPoints, Protobuf
    from ctrader_open_api.messages import OpenApiMessages_pb2 as OA
    from ctrader_open_api.messages import OpenApiModelMessages_pb2 as OAM
    from ctrader_open_api.messages import OpenApiCommonMessages_pb2 as OAC
    from twisted.internet import reactor

    ini = dt.datetime.strptime(date_from, "%Y-%m-%d")
    fim_d = dt.datetime.strptime(date_to, "%Y-%m-%d") + dt.timedelta(days=1)

    ctx = {
        "accountId": "", "symbolId": None, "digits": 5,
        "erro": None, "fim": False, "chamadas": 0,
        "fila": [], "atual": None, "buf": {},
    }

    client = Client(EndPoints.PROTOBUF_LIVE_HOST, EndPoints.PROTOBUF_PORT,
                    TcpProtocol)

    def heartbeat():
        if not ctx.get("fim"):
            try:
                client.send(OAC.ProtoHeartbeatEvent(),
                            responseTimeoutInSeconds=60).addErrback(lambda f: None)
            except Exception:
                pass
            reactor.callLater(10, heartbeat)

    def erro(f, fase):
        print(f"[ERROR {fase}]", repr(f))
        ctx["erro"] = fase
        fim()

    def on_erro(msg, fase):
        print(f"[ERROR {fase}]", Protobuf.extract(msg))
        ctx["erro"] = fase
        fim()

    # ---- flow: app auth -> account list -> account auth -> symbols -> ticks
    def passo_app_auth(_):
        req = OA.ProtoOAGetAccountListByAccessTokenReq()
        req.accessToken = access
        client.send(req, responseTimeoutInSeconds=20).addCallbacks(
            passo_account_list, lambda f: erro(f, "account_list"))

    def passo_account_list(msg):
        if msg.payloadType == OA.ProtoOAErrorRes().payloadType:
            return on_erro(msg, "account_list")
        res = Protobuf.extract(msg)
        contas = list(getattr(res, "ctidTraderAccount", []))
        live = [c for c in contas if getattr(c, "isLive", False)]
        escolhida = live[0] if live else contas[0]
        ctx["accountId"] = str(escolhida.ctidTraderAccountId)
        print(f"account: id={ctx['accountId']}")
        req = OA.ProtoOAAccountAuthReq()
        req.ctidTraderAccountId = int(ctx["accountId"])
        req.accessToken = access
        client.send(req, responseTimeoutInSeconds=20).addCallbacks(
            passo_symbols, lambda f: erro(f, "account_auth"))

    def passo_symbols(_):
        req = OA.ProtoOASymbolsListReq()
        req.ctidTraderAccountId = int(ctx["accountId"])
        req.includeArchivedSymbols = False
        client.send(req, responseTimeoutInSeconds=30).addCallbacks(
            passo_escolher_symbol, lambda f: erro(f, "symbols"))

    def passo_escolher_symbol(msg):
        if msg.payloadType == OA.ProtoOAErrorRes().payloadType:
            return on_erro(msg, "symbols")
        res = Protobuf.extract(msg)
        achou = None
        for s in list(getattr(res, "symbol", [])) + \
                 list(getattr(res, "archivedSymbol", [])):
            if getattr(s, "symbolName", "") == symbol:
                achou = s
                break
        if achou is None:
            print("Symbol not found:", symbol)
            fim()
            return
        ctx["symbolId"] = achou.symbolId
        ctx["digits"] = getattr(achou, "digits", 5)
        print(f"{symbol}: symbolId={ctx['symbolId']} digits={ctx['digits']}")

        # Empirical discovery (probe 2025-06-02): chunkSize=10000;
        # tickData NEWEST-FIRST; 1st timestamp=absolute ms, following ones
        # are negative ms deltas; 1st tick=absolute price x10^digits, the
        # following ones are price deltas. One EURUSD day ~= 100-250k
        # ticks/side -> queue of 1 item per day per side, flushed on
        # completion (bounded memory).
        dias = []
        d = ini
        while d < fim_d:
            if d.weekday() != 5:  # Saturday: forex closed, don't waste calls
                prox = min(d + dt.timedelta(days=1), fim_d)
                dias.append((_ts_ms(d), _ts_ms(prox) - 1))  # [d, prox)
            d += dt.timedelta(days=1)

        # resume: _progress.txt lists "YYYY-MM-DD SIDE" already written
        prog_path = os.path.join(out_dir, "_progress.txt")
        feitos = set()
        if os.path.exists(prog_path):
            with open(prog_path, "r", encoding="utf-8") as f:
                for ln in f:
                    p = ln.strip().split()
                    if len(p) == 2:
                        feitos.add((p[0], p[1]))
        ctx["prog_path"] = prog_path
        ctx["fila"] = []
        for a, b in dias:
            dia_str = dt.datetime.fromtimestamp(a / 1000, dt.timezone.utc).strftime("%Y-%m-%d")
            for t in ("BID", "ASK"):
                if (dia_str, t) not in feitos:
                    ctx["fila"].append((a, b, t))
        ctx["pulados"] = len(dias) * 2 - len(ctx["fila"])
        print(f"items to download: {len(ctx['fila'])} "
              f"({ctx['pulados']} already done)")
        ctx["atual"] = None
        ctx["buf"] = {}          # ts_ms -> price_int for current item
        ctx["tipo_item"] = None
        ctx["arq_bid"] = os.path.join(
            out_dir, f"{symbol}_ticks_{date_from}_{date_to}_BID.csv")
        ctx["arq_ask"] = os.path.join(
            out_dir, f"{symbol}_ticks_{date_from}_{date_to}_ASK.csv")
        ctx["gravados"] = {"BID": 0, "ASK": 0}
        if not probe:
            os.makedirs(out_dir, exist_ok=True)
            for k in ("bid", "ask"):
                path = ctx[f"arq_{k}"]
                # APPEND if it exists: opening with "w" truncated everything
                # on every resume while _progress.txt kept days marked =
                # SILENT DATA LOSS
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    continue
                with open(path, "w", encoding="utf-8", newline="") as f:
                    f.write("time_ms,time,price\n")
        proxima_pagina()

    # ---- tick pagination ----
    # Official docs (help.ctrader.com/open-api/messages):
    #   tickData arrives NEWEST-FIRST. The 1st timestamp is absolute ms; the
    #   following ones are ms DELTAS vs the previous tick (time decreases).
    #   Rebuild: t[0]=abs; t[i]=t[i-1]-delta[i].
    #   hasMore=true -> the response filled chunkSize and older ticks exist:
    #   request again with toTimestamp = oldest_received - 1.
    #   Official rate limit: 5 req/s for historical requests.
    def proxima_pagina():
        if ctx.get("fim"):
            return
        if ctx["chamadas"] >= max_requests:
            print("request limit reached")
            gravar_ou_relatar()
            return
        if ctx["atual"] is None:
            if not ctx["fila"]:
                gravar_ou_relatar()
                return
            ctx["atual"] = ctx["fila"].pop(0)
            ctx["tipo_item"] = ctx["atual"][2]
            ctx["buf"] = {}
        ini_usar, fim_usar, tipo = ctx["atual"]
        if ini_usar >= fim_usar:
            flush_item()
            ctx["atual"] = None
            proxima_pagina()
            return
        req = OA.ProtoOAGetTickDataReq()
        req.ctidTraderAccountId = int(ctx["accountId"])
        req.symbolId = ctx["symbolId"]
        req.type = getattr(OAM.ProtoOAQuoteType, tipo)
        req.fromTimestamp = ini_usar
        req.toTimestamp = fim_usar
        ctx["chamadas"] += 1
        ctx["req_ini"], ctx["req_fim"], ctx["tipo_req"] = ini_usar, fim_usar, tipo
        if ctx["chamadas"] <= 3:
            d1 = dt.datetime.fromtimestamp(ini_usar / 1000, dt.timezone.utc)
            d2 = dt.datetime.fromtimestamp(fim_usar / 1000, dt.timezone.utc)
            print(f"  req {tipo}: {d1} -> {d2}")
        client.send(req, responseTimeoutInSeconds=60).addCallbacks(
            passo_ticks, lambda f: erro(f, "tickdata"))

    def passo_ticks(msg):
        if msg.payloadType == OA.ProtoOAErrorRes().payloadType:
            ext = Protobuf.extract(msg)
            d1 = dt.datetime.fromtimestamp(ctx["req_ini"] / 1000, dt.timezone.utc)
            d2 = dt.datetime.fromtimestamp(ctx["req_fim"] / 1000, dt.timezone.utc)
            print(f"[ERROR tickdata {ctx.get('tipo_req')} {d1} -> {d2}]", ext)
            if "LIMIT" in str(ext).upper() or "429" in str(ext):
                print("  rate limit: waiting 10s and retrying")
                reactor.callLater(10, proxima_pagina)
                return
            # other error: retry the SAME request; only give up on the whole
            # day after 5 failures — never write a partial day or mark
            # progress
            ctx["erros_item"] = ctx.get("erros_item", 0) + 1
            if ctx["erros_item"] <= 5:
                print(f"  non-limit error: retry {ctx['erros_item']}/5 in 10s")
                reactor.callLater(10, proxima_pagina)
                return
            ctx.setdefault("buracos", []).append(
                (ctx["req_ini"], ctx["req_fim"], ctx["tipo_req"]))
            print(f"  [GAP] day discarded without writing: "
                  f"{ctx.get('tipo_req')} "
                  f"{dt.datetime.fromtimestamp(ctx['req_ini'] / 1000, dt.timezone.utc)} -> "
                  f"{dt.datetime.fromtimestamp(ctx['req_fim'] / 1000, dt.timezone.utc)}")
            ctx["buf"] = {}
            ctx["atual"] = None
            ctx["erros_item"] = 0
            proxima_pagina()
            return
        res = Protobuf.extract(msg)
        ctx["erros_item"] = 0
        tks = list(getattr(res, "tickData", []))
        tipo = ctx["tipo_req"]
        hasMore = getattr(res, "hasMore", False)
        if tks:
            # rebuild absolute times and absolute prices (newest-first)
            tempos = [tks[0].timestamp]
            precos = [tks[0].tick]
            for i in range(1, len(tks)):
                tempos.append(tempos[-1] + tks[i].timestamp)  # ms delta (<0)
                precos.append(precos[-1] + tks[i].tick)       # price delta
            if ctx["chamadas"] <= 3:
                print(f"    chunk {len(tks)} ticks, "
                      f"newest={dt.datetime.fromtimestamp(tempos[0] / 1000, dt.timezone.utc)}, "
                      f"oldest={dt.datetime.fromtimestamp(tempos[-1] / 1000, dt.timezone.utc)}, "
                      f"hasMore={hasMore}")
                for tp, pr in zip(tempos[:3], precos[:3]):
                    print(f"      ts={tp} price_int={pr} -> "
                          f"{pr / (10 ** ctx['digits']):.{ctx['digits']}f}")
            for tp_abs, pr in zip(tempos, precos):
                ctx["buf"][tp_abs] = pr
            if not hasMore:
                # hasMore=false -> ALL ticks of the interval: item complete
                flush_item()
                ctx["atual"] = None
            else:
                # next page: same from, to = oldest - 1
                ctx["atual"] = (ctx["atual"][0], tempos[-1] - 1, tipo)
                if ctx["atual"][0] >= ctx["atual"][1]:
                    flush_item()
                    ctx["atual"] = None
        else:
            # empty chunk: step toTimestamp back 1 day to avoid getting stuck
            ini0 = ctx["atual"][0]
            novo_to = ctx["atual"][1] - 86400000
            if novo_to <= ini0:
                flush_item()
                ctx["atual"] = None
            else:
                ctx["atual"] = (ini0, novo_to, tipo)
        if ctx["chamadas"] % 20 == 0:
            print(f"  {ctx['chamadas']} requests, "
                  f"{sum(ctx['gravados'].values())} ticks written, "
                  f"side={tipo}, hasMore={hasMore}")
        proxima_pagina()

    def flush_item():
        """Write the current item's buffer (1 day x side) to CSV and clear."""
        if probe or ctx["atual"] is None:
            ctx["buf"] = {}
            return
        tipo = ctx["tipo_item"]
        arq = ctx["arq_bid"] if tipo == "BID" else ctx["arq_ask"]
        escala = 10 ** ctx["digits"]
        with open(arq, "a", encoding="utf-8", newline="") as f:
            for ts_ in sorted(ctx["buf"]):
                d = dt.datetime.fromtimestamp(ts_ / 1000, dt.timezone.utc)
                f.write(f"{ts_},{d.strftime('%Y.%m.%d %H:%M:%S.%f')[:-3]},"
                        f"{ctx['buf'][ts_] / escala:.{ctx['digits']}f}\n")
        ctx["gravados"][tipo] += len(ctx["buf"])
        # checkpoint: mark the item as done for resume
        dia_str = dt.datetime.fromtimestamp(
            ctx["atual"][0] / 1000, dt.timezone.utc).strftime("%Y-%m-%d")
        with open(ctx["prog_path"], "a", encoding="utf-8") as f:
            f.write(f"{dia_str} {tipo}\n")
        ctx["buf"] = {}

    def gravar_ou_relatar():
        n = sum(ctx["gravados"].values())
        print(f"DONE: {ctx['chamadas']} requests, {n} ticks "
              f"(BID={ctx['gravados']['BID']}, ASK={ctx['gravados']['ASK']}), "
              f"queue left={len(ctx['fila'])}, gaps={len(ctx.get('buracos', []))}")
        if n == 0:
            print("RESULT: no ticks returned by the API for this period.")
        else:
            if ctx.get("buracos"):
                print(f"  gaps ({len(ctx['buracos'])}): {ctx['buracos'][:5]}")
            print(f"  BID -> {ctx['arq_bid']}")
            print(f"  ASK -> {ctx['arq_ask']}")
        fim()

    def fim():
        ctx["fim"] = True
        try:
            reactor.stop()
        except Exception:
            pass

    req = OA.ProtoOAApplicationAuthReq()
    req.clientId = env["OPEN_API_CLIENT_ID"]
    req.clientSecret = env["OPEN_API_SECRET"]
    client.send(req, responseTimeoutInSeconds=30).addCallbacks(
        lambda m: (on_erro(m, "app_auth")
                   if m.payloadType == OA.ProtoOAErrorRes().payloadType
                   else passo_app_auth(m)),
        lambda f: erro(f, "app_auth"))
    client.startService()
    client.setConnectedCallback(lambda c: print("connected"))
    client.setDisconnectedCallback(lambda c, r: print("disconnected:", r))
    heartbeat()
    reactor.run()
    if ctx.get("erro"):
        raise RuntimeError(f"download failed at stage: {ctx['erro']}")


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
