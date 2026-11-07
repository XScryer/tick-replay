"""Pure helpers of the downloader: no network and no cTrader package needed."""

from types import SimpleNamespace as NS

import pytest

from tickreplay import ctrader as ct


def test_decode_tick_chunk_newest_first_deltas():
    # absolute first entry, then negative time deltas and signed price deltas
    raw = [NS(timestamp=1_704_067_205_000, tick=110_432),
           NS(timestamp=-1_000, tick=-3),
           NS(timestamp=-250, tick=5),
           NS(timestamp=0, tick=-1)]          # two ticks in the same ms
    times, prices = ct.decode_tick_chunk(raw)
    assert times == [1_704_067_205_000, 1_704_067_204_000, 1_704_067_203_750,
                     1_704_067_203_750]
    assert prices == [110_432, 110_429, 110_434, 110_433]
    assert ct.decode_tick_chunk([]) == ([], [])


def test_format_price_uses_fixed_scale():
    assert ct.format_price(110_432, 5) == "1.10432"
    assert ct.format_price(14_812_345, 3) == "148.123"   # JPY pair, 3 digits
    assert ct.format_price(203_456_000, 2) == "2034.56"  # gold, 2 digits


def test_day_windows_inclusive_and_skip_saturday():
    w = ct.day_windows("2024-01-05", "2024-01-08")       # Fri..Mon
    assert [d for _, _, d in w] == ["2024-01-05", "2024-01-07", "2024-01-08"]
    a, b, _ = w[0]
    assert a == 1_704_412_800_000 and b - a == ct.DAY_MS - 1
    with pytest.raises(ValueError):
        ct.day_windows("2024-01-08", "2024-01-05")


def test_read_progress(tmp_path):
    p = tmp_path / "_progress.txt"
    p.write_text("2024-01-02 BID\n2024-01-02 ASK\n\ngarbage\n")
    assert ct.read_progress(str(p)) == {("2024-01-02", "BID"), ("2024-01-02", "ASK")}
    assert ct.read_progress(str(tmp_path / "missing.txt")) == set()


def _acc(ctid, login, live):
    return NS(ctidTraderAccountId=ctid, traderLogin=login, isLive=live)


def test_choose_account_by_type_and_login():
    accounts = [_acc(1, 5001, True), _acc(2, 9001, False), _acc(3, 9002, False)]
    assert ct.choose_account(accounts, live=False).ctidTraderAccountId == 2
    assert ct.choose_account(accounts, live=True).ctidTraderAccountId == 1
    assert ct.choose_account(accounts, live=False, account="9002").ctidTraderAccountId == 3
    assert ct.choose_account(accounts, live=False, account=3).ctidTraderAccountId == 3
    with pytest.raises(ValueError, match="does not match"):
        ct.choose_account(accounts, live=False, account="5001")
    with pytest.raises(ValueError, match="available"):
        ct.choose_account([_acc(1, 5001, True)], live=False)


def test_parse_redirect_uri():
    assert ct.parse_redirect_uri("http://localhost:8899/callback") == ("localhost", 8899)
    assert ct.parse_redirect_uri("http://127.0.0.1:17873/x") == ("127.0.0.1", 17873)
    with pytest.raises(ValueError):
        ct.parse_redirect_uri("https://example.com/callback")


def test_env_roundtrip_keeps_other_lines(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nOPEN_API_CLIENT_ID=abc\nOPEN_API_SECRET=xyz\n"
                   "CTRADER_ACCESS_TOKEN=\n")
    ct.save_env_tokens({"CTRADER_ACCESS_TOKEN": "tok", "CTRADER_REFRESH_TOKEN": "ref"},
                       str(env))
    text = env.read_text()
    assert text.startswith("# comment\nOPEN_API_CLIENT_ID=abc\n")
    assert "CTRADER_ACCESS_TOKEN=tok\n" in text and "CTRADER_REFRESH_TOKEN=ref\n" in text
    for k in ct.REQUIRED_ENV + ct.TOKEN_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPEN_API_SECRET", "from-env")
    loaded = ct.load_env(str(env))
    assert loaded["OPEN_API_CLIENT_ID"] == "abc"
    assert loaded["OPEN_API_SECRET"] == "from-env"     # real env wins
    assert loaded["CTRADER_ACCESS_TOKEN"] == "tok"


def test_token_dict_hides_response_on_error():
    with pytest.raises(RuntimeError) as e:
        ct._token_dict({"errorCode": "ACCESS_DENIED", "refreshToken": "secret!"})
    assert "secret!" not in str(e.value)
    tok = ct._token_dict({"accessToken": "a", "refreshToken": "r", "expiresIn": 100})
    assert tok["CTRADER_ACCESS_TOKEN"] == "a" and int(tok["CTRADER_TOKEN_EXPIRES_AT"]) > 100


def test_ensure_tokens_uses_cache_and_refreshes(monkeypatch, tmp_path):
    env_path = str(tmp_path / ".env")
    calls = []
    monkeypatch.setattr(ct, "oauth_flow", lambda *a, **k: calls.append("oauth") or
                        {"CTRADER_ACCESS_TOKEN": "new"})
    monkeypatch.setattr(ct, "refresh_tokens", lambda env, r: calls.append(("refresh", r)) or
                        {"CTRADER_ACCESS_TOKEN": "refreshed"})
    fresh = {"CTRADER_ACCESS_TOKEN": "cached", "CTRADER_REFRESH_TOKEN": "r1",
             "CTRADER_TOKEN_EXPIRES_AT": str(2**40)}
    assert ct.ensure_tokens(fresh, env_path)["CTRADER_ACCESS_TOKEN"] == "cached"
    stale = dict(fresh, CTRADER_TOKEN_EXPIRES_AT="1")
    assert ct.ensure_tokens(stale, env_path)["CTRADER_ACCESS_TOKEN"] == "refreshed"
    assert ct.ensure_tokens({}, env_path)["CTRADER_ACCESS_TOKEN"] == "new"
    assert calls == [("refresh", "r1"), "oauth"]
