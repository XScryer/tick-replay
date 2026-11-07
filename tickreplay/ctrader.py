"""Download historical bid/ask ticks via the cTrader Open API (Spotware).

Why this exists: the cTrader Open API serves the raw tick history (bid and
ask separately) of the broker behind your cTrader ID — on demo accounts too.
Backtesting on those quotes means testing on the same spreads a live bot on
that broker would have seen.

Protocol notes (official docs: https://help.ctrader.com/open-api/):

* ``ProtoOAGetTickDataReq`` takes ``symbolId``, ``type`` (BID/ASK) and a
  ``fromTimestamp``/``toTimestamp`` window in epoch ms; a window may not
  exceed one week. This downloader requests one UTC day at a time, which
  also gives natural resume checkpoints.
* ``tickData`` comes NEWEST-FIRST and delta-encoded: the first entry holds
  an absolute timestamp and price, every following entry holds the
  difference to the previous one (timestamps decrease).
* Prices are integers in 1/100000 of a unit for every symbol
  (``price = tick / 100000``); the symbol's ``digits`` only says how many
  decimals are meaningful.
* ``hasMore=true`` means the chunk was full and older ticks exist in the
  window: request again with ``toTimestamp = oldest_received - 1``.
* Demo accounts must be served by ``demo.ctraderapi.com`` and live accounts
  by ``live.ctraderapi.com``; pick with ``--account-type``.

Usage::

    # 1. create an app at https://openapi.ctrader.com/apps and add the
    #    redirect URI below (default http://localhost:8899/callback)
    # 2. put the app credentials in .env (see .env.example)
    # 3. the first run opens a browser for OAuth; tokens are cached in .env
    python -m tickreplay.ctrader --symbol EURUSD --from 2024-01-01 --to 2024-12-31 --out data/EURUSD/raw

Interrupted runs resume where they stopped: just run the same command again.
Twisted's reactor cannot be restarted, so ``download_ticks`` can be called
only once per Python process.
"""

import argparse
import datetime as dt
import os
import time
from urllib.parse import urlparse

ENV_PATH = ".env"
REQUIRED_ENV = ("OPEN_API_CLIENT_ID", "OPEN_API_SECRET")
TOKEN_KEYS = ("CTRADER_ACCESS_TOKEN", "CTRADER_REFRESH_TOKEN",
              "CTRADER_TOKEN_EXPIRES_AT")
OPTIONAL_ENV = ("CTRADER_REDIRECT_URI",)
DEFAULT_REDIRECT_URI = "http://localhost:8899/callback"
PRICE_SCALE = 100_000          # Open API prices are in 1/100000 of a unit
DAY_MS = 86_400_000
TOKEN_ERRORS = ("CH_ACCESS_TOKEN_INVALID", "OA_AUTH_TOKEN_EXPIRED")
MAX_ITEM_RETRIES = 5
RETRY_DELAY_S = 10


# ---------------------------------------------------------------------------
# pure helpers (no network, unit-tested)
# ---------------------------------------------------------------------------
def load_env(path=ENV_PATH):
    """Read KEY=VALUE lines; real environment variables win over the file."""
    out = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    out[k.strip()] = v.strip()
    known = REQUIRED_ENV + TOKEN_KEYS + OPTIONAL_ENV
    out.update({k: v for k, v in os.environ.items() if k in known})
    return out


def save_env_tokens(tokens, path=ENV_PATH):
    """Persist OAuth tokens into .env, keeping every other line untouched."""
    lines = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]
    pending = {k: str(v) for k, v in tokens.items()}
    new_lines = []
    for ln in lines:
        if "=" in ln and not ln.strip().startswith("#"):
            k = ln.split("=", 1)[0].strip()
            if k in pending:
                new_lines.append(f"{k}={pending.pop(k)}")
                continue
        new_lines.append(ln)
    new_lines += [f"{k}={v}" for k, v in pending.items()]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")
    try:
        os.chmod(path, 0o600)       # tokens are credentials
    except OSError:
        pass                        # e.g. Windows / exotic filesystems


def parse_redirect_uri(uri):
    """Return (host, port) the local OAuth callback server must bind to."""
    u = urlparse(uri)
    if u.scheme != "http" or u.hostname not in ("localhost", "127.0.0.1"):
        raise ValueError(
            f"redirect URI must be http://localhost:<port>/... (got {uri!r})")
    return u.hostname, u.port or 80


def decode_tick_chunk(tick_data):
    """Decode a delta-encoded, newest-first ``tickData`` list.

    Returns ``(times_ms, prices_int)`` in the same newest-first order, with
    absolute values. Entries only need ``.timestamp`` and ``.tick``.
    """
    times, prices = [], []
    t = p = 0
    for i, item in enumerate(tick_data):
        if i == 0:
            t, p = item.timestamp, item.tick
        else:
            t += item.timestamp      # negative ms delta
            p += item.tick           # signed price delta
        times.append(t)
        prices.append(p)
    return times, prices


def day_windows(date_from, date_to):
    """UTC day windows ``(start_ms, end_ms_inclusive, 'YYYY-MM-DD')``.

    Saturdays are skipped (spot FX is closed all day; Sunday's evening
    session is kept).
    """
    d = dt.datetime.strptime(date_from, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    end = (dt.datetime.strptime(date_to, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
           + dt.timedelta(days=1))
    if end <= d:
        raise ValueError("--to must be on or after --from")
    out = []
    while d < end:
        if d.weekday() != 5:
            a = int(d.timestamp() * 1000)
            out.append((a, a + DAY_MS - 1, d.strftime("%Y-%m-%d")))
        d += dt.timedelta(days=1)
    return out


def read_progress(path):
    """Set of ``(day, side)`` already written, from a _progress.txt file."""
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for ln in f:
                p = ln.split()
                if len(p) == 2:
                    done.add((p[0], p[1]))
    return done


def choose_account(accounts, live, account=None):
    """Pick the trading account to authorize.

    ``account`` may be the login number shown in cTrader (``traderLogin``)
    or the ``ctidTraderAccountId``. Without it, the first account of the
    requested type (demo/live) is used. Raises ValueError with the list of
    available accounts when nothing matches.
    """
    def label(a):
        kind = "live" if getattr(a, "isLive", False) else "demo"
        return f"{getattr(a, 'traderLogin', '?')} ({kind})"

    if account is not None:
        matches = [a for a in accounts
                   if str(account) in (str(getattr(a, "traderLogin", "")),
                                       str(a.ctidTraderAccountId))]
    else:
        matches = [a for a in accounts if bool(getattr(a, "isLive", False)) == live]
    if matches:
        chosen = matches[0]
        if bool(getattr(chosen, "isLive", False)) != live:
            raise ValueError(
                f"account {label(chosen)} does not match --account-type "
                f"{'live' if live else 'demo'}")
        return chosen
    available = ", ".join(label(a) for a in accounts) or "none"
    raise ValueError(f"no matching {'live' if live else 'demo'} account "
                     f"(available: {available})")


def format_price(price_int, digits):
    return f"{price_int / PRICE_SCALE:.{digits}f}"


# ---------------------------------------------------------------------------
# OAuth
# ---------------------------------------------------------------------------
def _token_dict(resp):
    if "accessToken" not in resp:
        # never echo the whole response: it may contain credentials
        msg = resp.get("description") or resp.get("errorCode") or "unknown error"
        raise RuntimeError(f"cTrader token request failed: {msg}")
    expires_in = int(resp.get("expiresIn") or 0)
    return {
        "CTRADER_ACCESS_TOKEN": resp["accessToken"],
        "CTRADER_REFRESH_TOKEN": resp.get("refreshToken", ""),
        "CTRADER_TOKEN_EXPIRES_AT": int(time.time()) + expires_in if expires_in else "",
    }


def oauth_flow(env, redirect_uri, timeout_s=600):
    """Interactive first-time auth: opens a browser, catches ?code= locally."""
    import webbrowser
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from threading import Thread
    from urllib.parse import parse_qs

    from ctrader_open_api import Auth

    auth = Auth(env["OPEN_API_CLIENT_ID"], env["OPEN_API_SECRET"], redirect_uri)
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

    host, port = parse_redirect_uri(redirect_uri)
    srv = HTTPServer((host, port), Handler)
    Thread(target=srv.serve_forever, daemon=True).start()
    url = auth.getAuthUri()
    print("Open this URL in your browser and approve access:\n ", url)
    webbrowser.open(url)
    t0 = time.time()
    while "code" not in result and time.time() - t0 < timeout_s:
        time.sleep(0.5)
    srv.shutdown()
    if "code" not in result:
        raise RuntimeError(f"OAuth timed out after {timeout_s // 60} minutes")
    return _token_dict(auth.getToken(result["code"]))


def refresh_tokens(env, refresh_token):
    from ctrader_open_api import Auth
    auth = Auth(env["OPEN_API_CLIENT_ID"], env["OPEN_API_SECRET"],
                env.get("CTRADER_REDIRECT_URI") or DEFAULT_REDIRECT_URI)
    return _token_dict(auth.refreshToken(refresh_token))


def ensure_tokens(env, env_path, reauth=False):
    """Return a usable token dict: cached, refreshed if near expiry, or new."""
    redirect_uri = env.get("CTRADER_REDIRECT_URI") or DEFAULT_REDIRECT_URI
    tokens = {k: env[k] for k in TOKEN_KEYS if env.get(k)}
    if reauth or not tokens.get("CTRADER_ACCESS_TOKEN"):
        tokens = oauth_flow(env, redirect_uri)
        save_env_tokens(tokens, env_path)
        return tokens
    expires_at = int(tokens.get("CTRADER_TOKEN_EXPIRES_AT") or 0)
    if expires_at and expires_at - time.time() < 86_400 and \
            tokens.get("CTRADER_REFRESH_TOKEN"):
        print("access token expires soon: refreshing")
        tokens = refresh_tokens(env, tokens["CTRADER_REFRESH_TOKEN"])
        save_env_tokens(tokens, env_path)
    return tokens


# ---------------------------------------------------------------------------
# downloader
# ---------------------------------------------------------------------------
class TickDownloader:
    """Event-driven download: app auth -> accounts -> account auth ->
    symbol lookup -> one request queue item per (UTC day, side)."""

    def __init__(self, env, tokens, symbol, date_from, date_to, out_dir,
                 live=False, account=None, probe=False, max_requests=100_000,
                 env_path=ENV_PATH):
        from ctrader_open_api import Client, EndPoints, TcpProtocol

        self.env, self.tokens, self.env_path = env, tokens, env_path
        self.symbol, self.date_from, self.date_to = symbol, date_from, date_to
        self.out_dir, self.live, self.account = out_dir, live, account
        self.probe, self.max_requests = probe, max_requests
        host = EndPoints.PROTOBUF_LIVE_HOST if live else EndPoints.PROTOBUF_DEMO_HOST
        print(f"host: {host}")
        # the client itself throttles to 5 messages/s, the documented limit
        # for historical-data requests
        self.client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)
        self.account_id = None
        self.symbol_id = None
        self.digits = 5
        self.error = None
        self.done = False
        self.refreshed = False
        self.started = False
        self.requests = 0
        self.queue = []
        self.current = None        # (win_start, win_end, side, day)
        self.item_times, self.item_prices = [], []
        self.item_errors = 0
        self.gaps = []
        self.written = {"BID": 0, "ASK": 0}
        self.files = {s: os.path.join(out_dir, f"{symbol}_ticks_{date_from}_{date_to}_{s}.csv")
                      for s in ("BID", "ASK")}
        self.prog_path = os.path.join(out_dir, "_progress.txt")

    # -- plumbing ---------------------------------------------------------
    def _oa(self):
        from ctrader_open_api.messages import OpenApiMessages_pb2 as OA
        return OA

    def _send(self, req, on_ok, stage, timeout=30):
        def ok(msg):
            if msg.payloadType == self._oa().ProtoOAErrorRes().payloadType:
                return self._on_api_error(msg, stage, on_ok, req)
            return on_ok(msg)
        self.client.send(req, responseTimeoutInSeconds=timeout).addCallbacks(
            ok, lambda f: self._fail(stage, f.getErrorMessage()))

    def _on_api_error(self, msg, stage, on_ok, req):
        from ctrader_open_api import Protobuf
        from twisted.internet import reactor
        err = Protobuf.extract(msg)
        code = getattr(err, "errorCode", "")
        desc = getattr(err, "description", "")
        if code in TOKEN_ERRORS and not self.refreshed and \
                self.tokens.get("CTRADER_REFRESH_TOKEN"):
            print(f"[{stage}] access token rejected ({code}): refreshing")
            self.refreshed = True
            try:
                self.tokens = refresh_tokens(self.env, self.tokens["CTRADER_REFRESH_TOKEN"])
            except RuntimeError as e:
                return self._fail(stage, f"{e} — run again with --reauth")
            save_env_tokens(self.tokens, self.env_path)
            return self._step_account_list(None)
        if code in TOKEN_ERRORS:
            return self._fail(stage, f"{code}: run again with --reauth")
        if stage == "tickdata":
            return self._on_tick_error(code, desc)
        if code == "REQUEST_FREQUENCY_EXCEEDED":
            print(f"[{stage}] rate limited: retrying in {RETRY_DELAY_S}s")
            reactor.callLater(RETRY_DELAY_S, self._send, req, on_ok, stage)
            return None
        return self._fail(stage, f"{code} {desc}".strip())

    def _fail(self, stage, detail):
        print(f"[ERROR {stage}] {detail}")
        self.error = f"{stage}: {detail}"
        self._shutdown()

    def _shutdown(self):
        from twisted.internet import reactor
        if self.done:
            return
        self.done = True
        if reactor.running:
            reactor.stop()

    def _heartbeat(self):
        from ctrader_open_api.messages import OpenApiCommonMessages_pb2 as OAC
        from twisted.internet import reactor
        if self.done:
            return
        self.client.send(OAC.ProtoHeartbeatEvent(),
                         responseTimeoutInSeconds=60).addErrback(lambda f: None)
        reactor.callLater(10, self._heartbeat)

    # -- flow -------------------------------------------------------------
    def run(self):
        from twisted.internet import reactor
        OA = self._oa()
        req = OA.ProtoOAApplicationAuthReq()
        req.clientId = self.env["OPEN_API_CLIENT_ID"]
        req.clientSecret = self.env["OPEN_API_SECRET"]
        self._send(req, self._step_account_list, "app_auth")
        self.client.setConnectedCallback(lambda c: print("connected"))
        self.client.setDisconnectedCallback(self._on_disconnect)
        self.client.startService()
        self._heartbeat()
        reactor.run()
        if self.error:
            raise RuntimeError(f"download failed at {self.error}")

    def _on_disconnect(self, _client, reason):
        if not self.done:
            # authentication state is lost on reconnect; stop cleanly and let
            # the user re-run (progress is checkpointed per day)
            self._fail("connection", f"disconnected ({reason}); re-run to resume")

    def _step_account_list(self, _msg):
        req = self._oa().ProtoOAGetAccountListByAccessTokenReq()
        req.accessToken = self.tokens["CTRADER_ACCESS_TOKEN"]
        self._send(req, self._step_account_auth, "account_list")

    def _step_account_auth(self, msg):
        from ctrader_open_api import Protobuf
        accounts = list(getattr(Protobuf.extract(msg), "ctidTraderAccount", []))
        try:
            chosen = choose_account(accounts, self.live, self.account)
        except ValueError as e:
            return self._fail("account_list", str(e))
        self.account_id = int(chosen.ctidTraderAccountId)
        kind = "live" if self.live else "demo"
        print(f"account: login={getattr(chosen, 'traderLogin', '?')} ({kind})")
        req = self._oa().ProtoOAAccountAuthReq()
        req.ctidTraderAccountId = self.account_id
        req.accessToken = self.tokens["CTRADER_ACCESS_TOKEN"]
        self._send(req, self._step_symbols, "account_auth")
        return None

    def _step_symbols(self, _msg):
        req = self._oa().ProtoOASymbolsListReq()
        req.ctidTraderAccountId = self.account_id
        self._send(req, self._step_symbol_details, "symbols")

    def _step_symbol_details(self, msg):
        from ctrader_open_api import Protobuf
        res = Protobuf.extract(msg)
        found = next((s for s in res.symbol if s.symbolName == self.symbol), None)
        if found is None:
            return self._fail("symbols", f"symbol {self.symbol!r} not found on this account")
        self.symbol_id = found.symbolId
        req = self._oa().ProtoOASymbolByIdReq()
        req.ctidTraderAccountId = self.account_id
        req.symbolId.append(self.symbol_id)
        self._send(req, self._step_start, "symbol_by_id")
        return None

    def _step_start(self, msg):
        from ctrader_open_api import Protobuf
        if self.started:              # re-authenticated after a token refresh
            return self._next_page()
        self.started = True
        res = Protobuf.extract(msg)
        if res.symbol:
            self.digits = int(res.symbol[0].digits)
        print(f"{self.symbol}: symbolId={self.symbol_id} digits={self.digits}")

        done = read_progress(self.prog_path)
        for a, b, day in day_windows(self.date_from, self.date_to):
            for side in ("BID", "ASK"):
                if (day, side) not in done:
                    self.queue.append((a, b, side, day))
        total = len(day_windows(self.date_from, self.date_to)) * 2
        print(f"items to download: {len(self.queue)} ({total - len(self.queue)} already done)")
        if not self.probe:
            os.makedirs(self.out_dir, exist_ok=True)
            for path in self.files.values():
                # append on resume: truncating here would silently drop days
                # that _progress.txt already marks as done
                if not (os.path.exists(path) and os.path.getsize(path) > 0):
                    with open(path, "w", encoding="utf-8", newline="") as f:
                        f.write("time_ms,time,price\n")
        self._next_page()

    # -- pagination -------------------------------------------------------
    def _next_page(self):
        from ctrader_open_api.messages import OpenApiModelMessages_pb2 as OAM
        if self.done:
            return
        if self.requests >= self.max_requests:
            print("request limit reached")
            return self._finish()
        if self.current is None:
            if not self.queue:
                return self._finish()
            self.current = self.queue.pop(0)
            self.item_times, self.item_prices = [], []
        win_start, win_end, side, _day = self.current
        req = self._oa().ProtoOAGetTickDataReq()
        req.ctidTraderAccountId = self.account_id
        req.symbolId = self.symbol_id
        req.type = getattr(OAM.ProtoOAQuoteType, side)
        req.fromTimestamp = win_start
        req.toTimestamp = win_end
        self.requests += 1
        self._send(req, self._on_ticks, "tickdata", timeout=60)
        return None

    def _on_ticks(self, msg):
        from ctrader_open_api import Protobuf
        res = Protobuf.extract(msg)
        self.item_errors = 0
        times, prices = decode_tick_chunk(list(res.tickData))
        win_start, _win_end, side, day = self.current
        self.item_times += times
        self.item_prices += prices
        if res.hasMore and times and times[-1] - 1 >= win_start:
            self.current = (win_start, times[-1] - 1, side, day)
        else:
            self._flush_item()
            self.current = None
        if self.requests % 50 == 0:
            print(f"  {self.requests} requests, {sum(self.written.values())} ticks written "
                  f"(at {day} {side})")
        self._next_page()

    def _on_tick_error(self, code, desc):
        from twisted.internet import reactor
        _a, _b, side, day = self.current
        print(f"[tickdata {day} {side}] {code} {desc}".rstrip())
        self.item_errors += 1
        if code == "REQUEST_FREQUENCY_EXCEEDED" or self.item_errors <= MAX_ITEM_RETRIES:
            print(f"  retry {self.item_errors}/{MAX_ITEM_RETRIES} in {RETRY_DELAY_S}s")
            # retry the same page; nothing partial is ever written
            reactor.callLater(RETRY_DELAY_S, self._next_page)
            return
        print(f"  [GAP] {day} {side} skipped (not marked done; re-run to retry)")
        self.gaps.append((day, side))
        self.current = None
        self.item_errors = 0
        self._next_page()

    def _flush_item(self):
        """Write the finished (day, side) item and checkpoint it."""
        _a, _b, side, day = self.current
        n = len(self.item_times)
        if not self.probe:
            with open(self.files[side], "a", encoding="utf-8", newline="") as f:
                # pages arrive newest-first: reverse to chronological order
                for t, p in zip(reversed(self.item_times), reversed(self.item_prices)):
                    stamp = dt.datetime.fromtimestamp(t / 1000, dt.timezone.utc)
                    f.write(f"{t},{stamp.strftime('%Y.%m.%d %H:%M:%S.%f')[:-3]},"
                            f"{format_price(p, self.digits)}\n")
            with open(self.prog_path, "a", encoding="utf-8") as f:
                f.write(f"{day} {side}\n")
        self.written[side] += n
        self.item_times, self.item_prices = [], []

    def _finish(self):
        n = sum(self.written.values())
        print(f"DONE: {self.requests} requests, {n} ticks "
              f"(BID={self.written['BID']}, ASK={self.written['ASK']}), "
              f"gaps={len(self.gaps)}")
        if self.gaps:
            print(f"  gaps: {self.gaps[:10]}")
        if n and not self.probe:
            print(f"  BID -> {self.files['BID']}\n  ASK -> {self.files['ASK']}")
        elif not n:
            print("no ticks returned for this period (check symbol/date range)")
        self._shutdown()


def download_ticks(symbol, date_from, date_to, out_dir, live=False, account=None,
                   probe=False, max_requests=100_000, env_path=ENV_PATH,
                   reauth=False):
    """Download BID+ASK ticks for ``symbol`` between two dates (inclusive).

    Writes ``<symbol>_ticks_<from>_<to>_<SIDE>.csv`` (columns
    ``time_ms,time,price``) plus a ``_progress.txt`` checkpoint into
    ``out_dir``. ``probe=True`` counts ticks without writing anything.
    """
    day_windows(date_from, date_to)          # validate dates before any I/O
    env = load_env(env_path)
    missing = [k for k in REQUIRED_ENV if not env.get(k)]
    if missing:
        raise RuntimeError(f"missing {missing}: copy .env.example to .env and "
                           f"fill in your cTrader Open API app credentials")
    tokens = ensure_tokens(env, env_path, reauth=reauth)
    TickDownloader(env, tokens, symbol, date_from, date_to, out_dir, live=live,
                   account=account, probe=probe, max_requests=max_requests,
                   env_path=env_path).run()


def _cli():
    ap = argparse.ArgumentParser(
        prog="tickreplay.ctrader",
        description="Download historical bid/ask ticks via the cTrader Open API")
    ap.add_argument("--symbol", required=True, help="symbol name as in cTrader, e.g. EURUSD")
    ap.add_argument("--from", dest="date_from", required=True, help="start date YYYY-MM-DD (UTC)")
    ap.add_argument("--to", dest="date_to", required=True, help="end date YYYY-MM-DD (UTC, inclusive)")
    ap.add_argument("--out", required=True, help="output dir for raw CSVs (one dir per symbol)")
    ap.add_argument("--account-type", choices=("demo", "live"), default="demo",
                    help="which kind of account to use (selects the API host)")
    ap.add_argument("--account", default=None,
                    help="account login number (default: first account of --account-type)")
    ap.add_argument("--probe", action="store_true", help="count ticks without writing files")
    ap.add_argument("--max-requests", type=int, default=100_000)
    ap.add_argument("--env", default=ENV_PATH, help="path to the .env file")
    ap.add_argument("--reauth", action="store_true",
                    help="ignore cached tokens and run the browser login again")
    args = ap.parse_args()
    try:
        download_ticks(args.symbol, args.date_from, args.date_to, args.out,
                       live=args.account_type == "live", account=args.account,
                       probe=args.probe, max_requests=args.max_requests,
                       env_path=args.env, reauth=args.reauth)
    except (RuntimeError, ValueError) as e:
        raise SystemExit(f"error: {e}") from None


if __name__ == "__main__":
    _cli()
