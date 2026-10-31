"""One chat call to whichever free model can take it now: failover across providers.

The models sit behind OpenAI-compatible endpoints (Groq, Google's Gemini API, OpenRouter), so one
small client (standard library only) talks to all of them. Providers are tried in the order of
config/llm.json. A provider is skipped while:
  * its key is missing (the key is read from the environment or from the project's .env file,
    never from the config, never logged);
  * it is cooling down after a 429 (rate limit) or 503: for its Retry-After, or until the next
    day's reset when the limit hit was a daily one;
  * the local counters say the call would break its per-minute or per-day limits (kept in
    out/llm/usage.json, so a restart does not forget what today used);
  * it answered 401/403/404 (wrong key or model): off for the rest of the run.
So when one model reaches its limit, the same call goes straight to the next one.

Only one account per provider: Google's API terms forbid getting around its limits ("You agree
to, and will not attempt to circumvent, such limitations", developers.google.com/terms) and its
quotas are per project anyway; OpenRouter's terms forbid "multiple accounts as a single user, for
purposes of bypassing ... use limits". The failover reaches the same goal within the rules.

    from cosim.llm import LLM
    llm = LLM()                                   # config/llm.json + .env
    r = llm.chat([{"role": "user", "content": "Say OK"}], max_tokens=16)
    print(r.text, r.provider, r.usage)
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("llm")
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "llm.json"
USAGE = ROOT / "out" / "llm" / "usage.json"


class NoProvider(RuntimeError):
    """Every provider is out (limits, keys, errors) for this call."""


@dataclass
class Provider:
    name: str
    base_url: str
    model: str
    key_env: str
    rpm: int = 10
    rpd: int = 200
    tpm: int = 0                  # 0: no per-minute token limit known
    json_schema: bool = True      # supports response_format json_schema
    tools: bool = True
    extra: dict = field(default_factory=dict)  # merged into the request body (e.g. reasoning_effort)
    timeout_s: float = 60.0


@dataclass
class Reply:
    text: str
    tool_calls: list
    provider: str
    model: str
    usage: dict
    seconds: float

    def json(self) -> dict:
        """The reply as JSON (models sometimes wrap it in ``` fences)."""
        t = self.text.strip()
        m = re.search(r"\{.*\}", t, re.DOTALL)
        return json.loads(m.group(0) if m else t)


def load_env(path: Path) -> None:
    """KEY=value lines into os.environ (values already set win). Comments with #."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def estimate_tokens(messages: list[dict], max_tokens: int) -> int:
    chars = sum(len(json.dumps(m.get("content", ""))) for m in messages)
    return chars // 4 + max_tokens


class LLM:
    def __init__(self, config: Path = CONFIG, usage_file: Path = USAGE, env_file: Path | None = None):
        for f in ([env_file] if env_file else [ROOT / ".env", ROOT / "config" / ".env"]):
            load_env(f)
        cfg = json.loads(Path(config).read_text(encoding="utf-8"))
        self.providers = [Provider(**p) for p in cfg["providers"]]
        self.usage_file = Path(usage_file)
        self._lock = threading.Lock()
        self._minute: dict[str, list[tuple[float, int]]] = {}  # provider -> [(time, tokens)]
        self._off: dict[str, str] = {}                          # provider -> why, for this run
        self.last_skipped: list[str] = []                       # why providers were passed over, last call
        self._state = self._load()

    # -- bookkeeping --------------------------------------------------------------

    def _load(self) -> dict:
        try:
            s = json.loads(self.usage_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            s = {}
        if s.get("day") != _day():
            s = {"day": _day(), "calls": {}, "tokens": {}, "cooldown_until": s.get("cooldown_until", {})}
        return s

    def _save(self) -> None:
        self.usage_file.parent.mkdir(parents=True, exist_ok=True)
        self.usage_file.write_text(json.dumps(self._state, indent=1), encoding="utf-8")

    def _key(self, p: Provider) -> str | None:
        return os.environ.get(p.key_env) or None

    def status(self) -> list[dict]:
        now = time.time()
        out = []
        for p in self.providers:
            cd = self._state["cooldown_until"].get(p.name, 0)
            out.append({"provider": p.name, "model": p.model, "key": bool(self._key(p)),
                        "calls_today": self._state["calls"].get(p.name, 0), "rpd": p.rpd,
                        "cooling_down_s": max(0, round(cd - now)), "off": self._off.get(p.name)})
        return out

    def _available(self, p: Provider, need_tokens: int) -> str | None:
        """None if p may take a call now, else why not."""
        if p.name in self._off:
            return self._off[p.name]
        if not self._key(p):
            return f"no key ({p.key_env})"
        now = time.time()
        if self._state["cooldown_until"].get(p.name, 0) > now:
            return "cooling down"
        if self._state["calls"].get(p.name, 0) >= p.rpd:
            return "daily limit reached (local count)"
        recent = [(t, n) for t, n in self._minute.get(p.name, []) if now - t < 60.0]
        self._minute[p.name] = recent
        if len(recent) >= p.rpm:
            return "per-minute requests"
        if p.tpm and sum(n for _, n in recent) + need_tokens > p.tpm:
            return "per-minute tokens"
        return None

    def _cool(self, p: Provider, seconds: float, why: str) -> None:
        self._state["cooldown_until"][p.name] = time.time() + seconds
        self._save()
        log.warning("%s: %s, cooling down %.0f s", p.name, why, seconds)

    # -- the call -----------------------------------------------------------------

    def chat(self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.2,
             tools: list[dict] | None = None, json_schema: dict | None = None, wait_s: float = 30.0) -> Reply:
        """Send to the first provider that can take it; on a limit or an outage, the next.
        json_schema: {"name": ..., "schema": {...}} for a structured reply. If every provider is
        busy only for the current minute, waits up to wait_s for one to free up."""
        need = estimate_tokens(messages, max_tokens)
        deadline = time.time() + wait_s
        errors = self.last_skipped = []
        while True:
            for p in self.providers:
                with self._lock:
                    why = self._available(p, need)
                    if why is None:
                        self._minute.setdefault(p.name, []).append((time.time(), need))
                if why is not None:
                    errors.append(f"{p.name}: {why}")
                    continue
                if tools and not p.tools:
                    errors.append(f"{p.name}: no tool calling")
                    continue
                r = self._call(p, messages, max_tokens, temperature, tools, json_schema, errors)
                if r is None:
                    continue
                if json_schema:
                    why = schema_errors(r, json_schema["schema"])
                    if why:  # a reply that breaks the schema counts as no reply: next provider
                        log.warning("%s: reply does not match the schema (%s)", p.name, why)
                        errors.append(f"{p.name}: reply off schema ({why})")
                        continue
                return r
            minute_only = all(e.endswith(("per-minute requests", "per-minute tokens")) or "cooling" in e
                              for e in errors[-len(self.providers):])
            if time.time() < deadline and minute_only:
                time.sleep(2.0)
                errors.clear()
                continue
            raise NoProvider("; ".join(errors[-len(self.providers) * 2:]))

    def _call(self, p: Provider, messages, max_tokens, temperature, tools, json_schema, errors) -> Reply | None:
        body = {"model": p.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        body.update(p.extra)
        if tools:
            body["tools"] = tools
        if json_schema:
            if p.json_schema and not tools:
                body["response_format"] = {"type": "json_schema", "json_schema": {**json_schema, "strict": True}}
            else:
                body["response_format"] = {"type": "json_object"}
                body["messages"] = [*messages, {"role": "system", "content":
                                    "Reply with one JSON object matching this schema: " + json.dumps(json_schema["schema"])}]
        r = self._post(p, body, errors)
        if r == "length":
            # Reasoning models (gpt-oss) spend part of max_tokens thinking; Groq then refuses a JSON
            # reply cut short (400). Once more with room to finish.
            body["max_tokens"] = min(4 * max_tokens, 8192)
            r = self._post(p, body, errors)
        if r == "schema" and json_schema:
            # Some endpoints refuse a strict schema, or reject a reply that breaks it (Groq: 400
            # json_validate_failed): ask the same model once more for plain JSON with the schema in the prompt.
            body["response_format"] = {"type": "json_object"}
            body["messages"] = [*messages, {"role": "system", "content":
                                "Reply with one JSON object matching this schema: " + json.dumps(json_schema["schema"])}]
            r = self._post(p, body, errors)
        return r if isinstance(r, Reply) else None

    def _post(self, p: Provider, body: dict, errors: list) -> Reply | str | None:
        """One request. A Reply; "length" for a 400 about max_tokens; "schema" for a 400 about the
        response format; else None."""
        req = urllib.request.Request(p.base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self._key(p)}",
                                              "Content-Type": "application/json", "Accept": "application/json",
                                              # Groq sits behind Cloudflare, which refuses Python's default
                                              # "Python-urllib/3.x" agent with 403 "error code: 1010"
                                              "User-Agent": "urban-traffic-carla/1.0"},
                                     method="POST")
        for attempt in range(2):
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=p.timeout_s) as resp:
                    data = json.loads(resp.read().decode())
                break
            except urllib.error.HTTPError as e:
                text = _safe(e.read().decode(errors="replace"))
                if e.code == 429:
                    self._cool(p, _retry_after(e.headers, text), "rate limit (429)")
                    errors.append(f"{p.name}: 429")
                    return None
                if e.code in (401, 403, 404):
                    self._off[p.name] = f"HTTP {e.code}: {text[:120]}"
                    log.error("%s: HTTP %d, off for this run: %s", p.name, e.code, text[:200])
                    errors.append(f"{p.name}: HTTP {e.code}")
                    return None
                if e.code >= 500 and attempt == 0:
                    time.sleep(1.5)
                    continue
                if e.code >= 500:
                    self._cool(p, 60.0, f"HTTP {e.code}")
                log.warning("%s: HTTP %d: %s", p.name, e.code, text[:300])
                errors.append(f"{p.name}: HTTP {e.code}: {text[:160]}")
                if e.code == 400 and re.search(r"max.?completion.?tokens|max_tokens|truncated", text, re.I):
                    return "length" if body["max_tokens"] < 8192 else None
                if e.code == 400 and "response_format" in body and re.search(r"json|schema|format", text, re.I):
                    return "schema"
                return None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == 0:
                    time.sleep(1.5)
                    continue
                self._cool(p, 60.0, f"unreachable ({e.__class__.__name__}: {_safe(str(e))[:120]})")
                errors.append(f"{p.name}: unreachable")
                return None
        msg = data["choices"][0]["message"]
        usage = data.get("usage") or {}
        with self._lock:
            self._state["calls"][p.name] = self._state["calls"].get(p.name, 0) + 1
            self._state["tokens"][p.name] = self._state["tokens"].get(p.name, 0) + int(usage.get("total_tokens", 0))
            self._save()
        log.info("%s: %s tokens in %.1f s", p.name, usage.get("total_tokens"), time.time() - t0)
        return Reply(msg.get("content") or "", msg.get("tool_calls") or [], p.name, p.model, usage, time.time() - t0)


def schema_errors(r: Reply, schema: dict) -> str | None:
    """None if the reply is JSON that fits the schema (types, required keys, array items), else why."""
    try:
        obj = r.json()
    except (ValueError, TypeError):
        return "not JSON"
    return _check(obj, schema, "$")


_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "integer": int, "number": (int, float)}


def _check(v, sc: dict, path: str) -> str | None:
    t = sc.get("type")
    if t and not isinstance(v, _TYPES.get(t, object)):
        return f"{path} is not {t}"
    if t == "object":
        for k in sc.get("required", []):
            if k not in v:
                return f"{path}.{k} missing"
        for k, sub in sc.get("properties", {}).items():
            if k in v and (why := _check(v[k], sub, f"{path}.{k}")):
                return why
    if t == "array" and "items" in sc:
        for i, x in enumerate(v):
            if why := _check(x, sc["items"], f"{path}[{i}]"):
                return why
    return None


def _day() -> str:
    # Groq and OpenRouter reset daily counts at UTC midnight, Google at midnight Pacific; the
    # local count is only a guard, the providers' 429s are what count.
    return time.strftime("%Y-%m-%d", time.gmtime())


def _safe(text: str) -> str:
    """Error text without anything that looks like a key."""
    return re.sub(r"(sk|gsk|AIza|AQ\.)[-_.A-Za-z0-9]{12,}", "<key>", text)


def _retry_after(headers, text: str) -> float:
    """Seconds to wait after a 429: Retry-After, Groq's x-ratelimit-reset-*, Google's retryDelay;
    a daily limit cools down for an hour (it is tried again then, and 429s again until reset)."""
    for h in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        v = headers.get(h) if headers else None
        if v:
            s = _duration(v)
            if s is not None:
                return max(1.0, min(s, 3600.0))
    m = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', text)
    if m:
        return max(1.0, float(m.group(1)))
    if re.search(r"per.?day|daily|RPD|TPD|quota", text, re.IGNORECASE):
        return 3600.0
    return 20.0


def _duration(v: str) -> float | None:
    """'7.66s', '2m59.56s', '1h2m', '30' -> seconds."""
    v = v.strip()
    try:
        return float(v)
    except ValueError:
        pass
    total, found = 0.0, False
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", v):
        found = True
        total += float(num) * {"ms": 0.001, "h": 3600, "m": 60, "s": 1}[unit]
    return total if found else None
