# diagnose_llm_connectivity.py
"""
Connectivity + TLS + auth diagnostic for your LLM client environment.

Checks (in order):
1) Env & base URL sanity
2) DNS resolution
3) TCP connect to host:443 (or parsed port)
4) TLS handshake with proper CA bundle
5) Raw HTTP GET /v1/models via httpx (with API key)
6) OpenAI SDK call (models.list)

Usage:
  python diagnose_llm_connectivity.py

Relies on env vars your client already uses:
  LLM_API_BASE (default: https://api.openai.com/v1)
  LLM_API_KEY  (required)
Also inspects: HTTPS_PROXY, HTTP_PROXY, NO_PROXY, SSL_CERT_FILE, REQUESTS_CA_BUNDLE
"""

import os, sys, socket, ssl, json, textwrap
from urllib.parse import urlparse
import certifi

def eprint(*a, **k): print(*a, file=sys.stderr, **k)

def step(title):
    print(f"\n=== {title} ===")

def mask(s, keep=4):
    if not s: return ""
    return s[:keep] + "..." + s[-keep:] if len(s) > keep*2 else "*" * len(s)

def fail(msg, code=1):
    eprint(f"FAIL: {msg}")
    sys.exit(code)

def warn(msg):
    eprint(f"WARNING: {msg}")

def get_base():
    base = os.getenv("LLM_API_BASE", "https://api.openai.com/v1").strip()
    # Basic sanity
    if not base.startswith("http"):
        fail(f"LLM_API_BASE looks invalid: {base!r}")
    return base

def parse_host_port(base):
    u = urlparse(base)
    host = u.hostname or ""
    port = u.port or (443 if (u.scheme or "https").lower() == "https" else 80)
    path = u.path or "/v1"
    return host, port, path

def check_env():
    step("Environment overview")
    base = get_base()
    key = os.getenv("LLM_API_KEY", "")
    https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    http_proxy  = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
    no_proxy    = os.getenv("NO_PROXY") or os.getenv("no_proxy")
    ssl_cert    = os.getenv("SSL_CERT_FILE")
    reqs_ca     = os.getenv("REQUESTS_CA_BUNDLE")

    print("LLM_API_BASE      :", base)
    print("LLM_API_KEY       :", mask(key))
    print("HTTPS_PROXY       :", https_proxy or "(unset)")
    print("HTTP_PROXY        :", http_proxy  or "(unset)")
    print("NO_PROXY          :", no_proxy    or "(unset)")
    print("SSL_CERT_FILE     :", ssl_cert    or "(unset)")
    print("REQUESTS_CA_BUNDLE:", reqs_ca     or "(unset)")
    print("certifi.where()   :", certifi.where())

    if not key:
        fail("LLM_API_KEY is not set")

    return base, key

def check_dns(host, port):
    step(f"DNS lookup for {host}")
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        addrs = sorted({i[4][0] for i in infos})
        print("Resolved addresses:", ", ".join(addrs))
        if not addrs:
            fail("DNS returned no addresses")
        return addrs
    except Exception as e:
        fail(f"DNS resolution failed: {e}")

def check_tcp(host, port, timeout=5.0):
    step(f"TCP connect to {host}:{port}")
    try:
        with socket.create_connection((host, port), timeout=timeout):
            print("TCP connect: OK")
    except Exception as e:
        fail(f"TCP connect failed: {e}")

def check_tls(host, port, timeout=10.0):
    step("TLS handshake")
    ctx = ssl.create_default_context(cafile=certifi.where())
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()
                subj = dict(x[0] for x in cert.get("subject", []))
                issuer = dict(x[0] for x in cert.get("issuer", []))
                print("TLS protocol:", ssock.version())
                print("Cert subject CN:", subj.get("commonName"))
                print("Cert issuer CN :", issuer.get("commonName"))
                print("TLS handshake: OK")
    except ssl.SSLCertVerificationError as e:
        fail(f"TLS cert verification failed: {e}")
    except Exception as e:
        fail(f"TLS handshake failed: {e}")

def check_http_models(base, key, timeout=15.0):
    step("HTTP GET /v1/models via httpx")
    try:
        import httpx
    except Exception:
        warn("httpx not installed; skipping raw HTTP check")
        return

    u = urlparse(base)
    # Ensure path ends with /v1
    api_base = f"{u.scheme}://{u.netloc}"
    path = u.path or "/v1"
    if not path.endswith("/v1"):
        # Try to find '/v1' prefix within the path; otherwise append
        if "/v1" in path:
            path = path[path.index("/v1"):]
        else:
            path = "/v1"
    url = api_base + path + "/models"

    try:
        r = httpx.get(
            url,
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
            follow_redirects=True,
        )
        print("Status:", r.status_code)
        if r.status_code == 200:
            print("HTTP check OK (received models list).")
        elif r.status_code in (401, 403):
            fail(f"HTTP reachable, but auth rejected (status {r.status_code}). Check API key / org access.", 2)
        else:
            fail(f"HTTP reachable, unexpected status {r.status_code}: {r.text[:200]}", 3)
    except httpx.ProxyError as e:
        fail(f"Proxy error: {e}")
    except httpx.ConnectTimeout as e:
        fail(f"HTTP connect timeout: {e}")
    except httpx.ConnectError as e:
        fail(f"HTTP connect error: {e}")
    except httpx.HTTPError as e:
        fail(f"HTTP error: {e}")

def check_sdk(base, key):
    step("OpenAI SDK models.list()")
    try:
        from openai import OpenAI
        client = OpenAI(base_url=base, api_key=key, timeout=30.0, max_retries=2)
        models = client.models.list()
        # Don’t dump everything; print first id
        mid = getattr(models.data[0], "id", None) if getattr(models, "data", None) else None
        print("SDK reachable. First model id:", mid or "(received list)")
        print("\nAll checks PASS.")
        return
    except Exception as e:
        fail(f"OpenAI SDK call failed: {e}", 4)

def main():
    base, key = check_env()
    host, port, _ = parse_host_port(base)
    if not host:
        fail(f"Could not parse host from LLM_API_BASE={base!r}")
    check_dns(host, port)
    check_tcp(host, port)
    check_tls(host, port)
    check_http_models(base, key)
    check_sdk(base, key)

if __name__ == "__main__":
    main()
