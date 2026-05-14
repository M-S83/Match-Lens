"""
batch_runner.py — Anthropic Message Batches API runner for Match Lens

Submits all Tier 1 windows as a single batch. Polls for completion.
Writes results to the standard agent_logs/ file paths.

Key benefits vs sequential API calls:
  - 50% cost reduction on all batch requests
  - No Claude Code session required to stay open
  - All windows submitted at once, processed async by Anthropic
  - Survives network interruptions (poll until done)
  - Zero risk of credit exhaustion mid-window

Usage:
    from batch_runner import submit_batch, poll_batch, collect_results

    batch_id = submit_batch(match_dir, windows, prompts, step="3a")
    results  = poll_batch(batch_id, match_dir, poll_interval=60)
    collect_results(results, match_dir, step="3a")
"""

import json, os, time, anthropic
from datetime import datetime
from pipeline_state import mark_window, store_batch_id


MODEL       = "claude-sonnet-4-20250514"
MAX_TOKENS  = 4096
POLL_SECS   = 60       # poll every 60 seconds
MAX_POLLS   = 240      # 4 hours max wait

client = anthropic.Anthropic()


# ── Build batch requests ──────────────────────────────────────────────────────

def build_request(window_id: str, prompt: str, images: list,
                  step: str = "3a") -> dict:
    """
    Build a single batch request entry for one window.
    images: list of base64-encoded image dicts or file paths.
    """
    content = []

    # Add images first (vision input)
    for img in images:
        if isinstance(img, str) and os.path.exists(img):
            import base64
            with open(img, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            ext  = os.path.splitext(img)[1].lower().lstrip(".")
            mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "png": "image/png"}.get(ext, "image/jpeg")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": b64}
            })
        elif isinstance(img, dict):
            content.append(img)

    # Add text prompt
    content.append({"type": "text", "text": prompt})

    return {
        "custom_id": f"{window_id}_{step}",
        "params": {
            "model":      MODEL,
            "max_tokens": MAX_TOKENS,
            "messages":   [{"role": "user", "content": content}],
        }
    }


# ── Submit batch ──────────────────────────────────────────────────────────────

def _probe_api_health(client) -> None:
    """
    Send a minimal probe to /v1/messages to verify the API is
    reachable and credits are available before submitting a batch.

    Fix 35 B: when credits run out, /v1/messages/batches.create returns
    502/520/connection-error instead of a clean 400. Without this probe
    we burn ~10 minutes through 7 retries before failing. /v1/messages
    correctly returns 400 BadRequestError with a "credit balance too low"
    message, so we can detect billing issues immediately.
    """
    try:
        client.messages.create(
            model=MODEL,
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
    except anthropic.BadRequestError as e:
        # 400 from /v1/messages: check for billing keywords
        msg = str(e).lower()
        if "credit" in msg or "billing" in msg or "balance" in msg:
            raise RuntimeError(
                "Anthropic credit balance too low. "
                "Top up at https://console.anthropic.com/settings/billing"
            ) from e
        # Otherwise it's an unrelated validation error -- API is healthy.
    except anthropic.AuthenticationError as e:
        raise RuntimeError(f"API authentication failed: {e}") from e
    except Exception:
        # Anything else (5xx, connection error, timeout) -- API may be
        # genuinely degraded. Let submit_batch's with_retry handle it.
        pass


MAX_REQUESTS_PER_BATCH = 20  # Fix 36b: ~200MB cap at typical 30-frame requests
                              # (Anthropic Batches API hard limit is 256MB)


def submit_batch(match_dir: str, state: dict,
                 requests: list, step: str) -> str:
    """
    Submit a list of request dicts to the Batches API.
    Returns the batch_id (or comma-separated batch_ids for chunked submissions).

    Fix 36b: long matches (>20 windows, e.g. extra-time matches) push the
    batch payload past the 256MB API limit. Auto-chunk into <=20-request
    batches and return a comma-separated composite id. poll_batch and
    collect_results detect this and iterate.
    """
    # Fix 35 B: fail fast on billing issues -- otherwise the batch endpoint
    # masks them as transient connection errors and the retry loop wastes
    # ~10 minutes before giving up.
    _probe_api_health(client)

    if len(requests) <= MAX_REQUESTS_PER_BATCH:
        print(f"  [BATCH] Submitting {len(requests)} requests for step {step}...")
        batch = client.messages.batches.create(requests=requests)
        batch_id = batch.id
        print(f"  [BATCH] Submitted: {batch_id}")
        print(f"  [BATCH] Processing status: {batch.processing_status}")
        state = store_batch_id(match_dir, state, f"{step}_batch_id", batch_id)
        return batch_id

    # Chunked path: split into <=MAX_REQUESTS_PER_BATCH chunks.
    chunks = [requests[i:i + MAX_REQUESTS_PER_BATCH]
              for i in range(0, len(requests), MAX_REQUESTS_PER_BATCH)]
    print(f"  [BATCH] Splitting {len(requests)} requests into {len(chunks)} "
          f"batches (max {MAX_REQUESTS_PER_BATCH} per batch) for step {step}...")

    batch_ids = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  [BATCH] Submitting chunk {i}/{len(chunks)} "
              f"({len(chunk)} requests)...")
        batch = client.messages.batches.create(requests=chunk)
        batch_ids.append(batch.id)
        print(f"  [BATCH]   chunk {i} -> {batch.id} "
              f"(status: {batch.processing_status})")

    composite = ",".join(batch_ids)
    state = store_batch_id(match_dir, state, f"{step}_batch_id", composite)
    return composite


# ── Exponential backoff wrapper ───────────────────────────────────────────────
# Defined before poll_batch / collect_results so the file reads top-to-bottom.

def with_retry(fn, max_retries: int = 7, base_delay: float = 10.0):
    """
    Wrap any API call with exponential backoff on rate-limit, server (5xx)
    and transient network errors (connection reset, timeout).

    Fix 34a follow-up #2: retry on ANY 5xx, not just (502, 503, 504, 529).
    Cloudflare 520-526 are all explicitly retryable (per the error payload),
    and a generic 500 is by definition server-side. 4xx errors still raise
    immediately (auth / bad request etc.).

    Defaults bumped to 7 retries / 10s base so a real gateway storm
    (multiple consecutive 5xxs from Cloudflare) is survivable. Worst case
    waits: 10, 20, 40, 80, 160, 320, 640s -> ~21 min total.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except anthropic.RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"  [RETRY] Rate limit hit. Waiting {delay:.0f}s "
                  f"(attempt {attempt+1}/{max_retries})...")
            time.sleep(delay)
        except anthropic.APIStatusError as e:
            # Retry on any 5xx (transient server-side), bail on 4xx.
            if 500 <= (e.status_code or 0) < 600 and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"  [RETRY] Transient upstream error ({e.status_code}). "
                      f"Waiting {delay:.0f}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(delay)
            else:
                raise
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"  [RETRY] Connection error ({type(e).__name__}). "
                  f"Waiting {delay:.0f}s (attempt {attempt+1}/{max_retries})...")
            time.sleep(delay)


# ── Poll for completion ───────────────────────────────────────────────────────

def poll_batch(batch_id: str, poll_interval: int = POLL_SECS,
               max_polls: int = MAX_POLLS) -> object:
    """
    Poll until the batch (or all batches in a composite id) are complete.
    Returns the last completed batch object (or list when composite).

    Fix 36b: accepts comma-separated composite batch_id from chunked submits.
    Polls each chunk in turn until all reach terminal status.
    """
    if "," in batch_id:
        ids = batch_id.split(",")
        print(f"\n  [BATCH] Polling {len(ids)} chunked batches: {ids}")
        return [_poll_single(bid, poll_interval, max_polls) for bid in ids]
    return _poll_single(batch_id, poll_interval, max_polls)


def _poll_single(batch_id: str, poll_interval: int, max_polls: int) -> object:
    print(f"\n  [BATCH] Polling {batch_id}...")
    print(f"  [BATCH] Will poll every {poll_interval}s (max {max_polls} polls = "
          f"{max_polls * poll_interval // 3600}h)")

    for poll_n in range(max_polls):
        # Fix 34a P1.1: retrieve wrapped with_retry to survive transient
        # connection drops (WinError 10054) during the polling phase.
        batch = with_retry(lambda: client.messages.batches.retrieve(batch_id))
        status = batch.processing_status

        counts = batch.request_counts
        done   = getattr(counts, "succeeded", 0) + getattr(counts, "errored", 0)
        total  = getattr(counts, "processing", 0) + done

        print(f"  [BATCH] Poll {poll_n+1}: {status} — "
              f"{done}/{total} complete "
              f"({getattr(counts, 'succeeded', 0)} OK, "
              f"{getattr(counts, 'errored', 0)} errors) "
              f"[{datetime.now().strftime('%H:%M:%S')}]")

        # Fix 34a P1.3: exit on terminal statuses, not just "ended".
        # Without this, an out-of-band cancel (Anthropic console) would
        # poll uselessly for the full max_polls window.
        if status in ("ended", "cancelled", "canceled"):
            if status in ("cancelled", "canceled"):
                print(f"  [WARN] Batch {batch_id} was cancelled — "
                      f"results may be incomplete")
            else:
                print(f"  [BATCH] Complete.")
            return batch

        time.sleep(poll_interval)

    raise TimeoutError(f"Batch {batch_id} did not complete within "
                       f"{max_polls * poll_interval}s")


# ── Collect results ───────────────────────────────────────────────────────────

def collect_results(batch_id: str, match_dir: str,
                    state: dict, step: str) -> dict:
    """
    Iterate batch results and write each to the standard agent_logs path.
    Returns dict of {window_id: output_path or error}.

    Fix 36b: handles comma-separated composite batch_id from chunked submits
    by collecting from each chunk and merging the per-window results dict.
    """
    agent_logs = os.path.join(match_dir, "agent_logs")
    os.makedirs(agent_logs, exist_ok=True)

    results = {}
    suffix_map = {
        "3a": "structural",
        "3b": "player",
        "3d_event": "event",
        "3d_setpiece": "setpiece",
        "3d_recovery": "recovery",
    }
    suffix = suffix_map.get(step, step)

    # Fix 36b: split composite batch_id into individual chunks. Each chunk's
    # results iterator is fetched independently with retry (same protection
    # as the single-batch path).
    batch_ids = batch_id.split(",") if "," in batch_id else [batch_id]
    if len(batch_ids) > 1:
        print(f"  [BATCH] Collecting from {len(batch_ids)} chunked batches...")

    all_results = []
    for bid in batch_ids:
        # Fix 34a P1.2: materialise the iterator inside with_retry so a transient
        # network drop mid-iteration is caught here rather than crashing partway
        # through processing. Anthropic stores results 24h, so a re-fetch is safe.
        chunk_iter = with_retry(
            lambda b=bid: list(client.messages.batches.results(b))
        )
        all_results.extend(chunk_iter)

    for result in all_results:
        custom_id = result.custom_id
        window_id = custom_id.replace(f"_{step}", "")

        if result.result.type == "succeeded":
            message = result.result.message
            # Extract text content
            text_blocks = [c.text for c in message.content
                           if hasattr(c, "text")]
            raw_text = "\n".join(text_blocks)

            # Parse JSON output
            try:
                import re as _re
                clean = raw_text.strip()
                # Model often prepends prose before the JSON block.
                # Strategy: find the first ``` fence or the first { and extract from there.
                _fence = _re.search(r'```(?:json)?\s*\n?([\s\S]*?)\s*```', clean)
                if _fence:
                    clean = _fence.group(1).strip()
                else:
                    # No fence — find first { ... last } (handles prose prefix)
                    _js = clean.find('{')
                    _je = clean.rfind('}')
                    if _js != -1 and _je > _js:
                        clean = clean[_js:_je + 1]
                output = json.loads(clean)
            except json.JSONDecodeError as e:
                output = {"raw_text": raw_text, "parse_error": str(e)}
                mark_window(match_dir, state, window_id, step, "failed",
                            f"JSON parse error: {e}")
                results[window_id] = {"status": "parse_error", "error": str(e)}
                continue

            # Write to agent_logs
            out_path = os.path.join(agent_logs,
                f"agent_{window_id}_{suffix}.json")
            # Fix 34a P2.2: guard the file write -- disk full, permission
            # denied, AV scanner lock (Windows) etc. should fail this window
            # rather than crash the whole collection loop.
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(output, f, indent=2)
            except OSError as _ioe:
                print(f"  [ERROR] Failed to write {out_path}: {_ioe}")
                mark_window(match_dir, state, window_id, step, "failed",
                            f"file_write_error: {_ioe}")
                results[window_id] = {"status": "failed",
                                       "error": f"file_write_error: {_ioe}"}
                continue

            mark_window(match_dir, state, window_id, step, "complete")
            results[window_id] = {"status": "complete", "path": out_path}

        else:
            # Fix 34a P2.1: typed error messages by result.result.type so
            # "expired" doesn't log as "None" and "cancelled" is identifiable.
            _rtype = getattr(result.result, "type", "unknown")
            if _rtype == "expired":
                error_msg = "Batch result expired (>24h since submission)"
            elif _rtype in ("cancelled", "canceled"):
                error_msg = "Batch was cancelled before this result completed"
            elif _rtype == "errored":
                _err = getattr(result.result, "error", None)
                error_msg = str(_err) if _err else "Unknown error (no detail returned)"
            else:
                error_msg = f"Unexpected result type: {_rtype}"
            mark_window(match_dir, state, window_id, step, "failed", error_msg)
            results[window_id] = {"status": "failed", "error": error_msg}

    successes = sum(1 for r in results.values() if r["status"] == "complete")
    failures  = sum(1 for r in results.values() if r["status"] != "complete")
    print(f"  [BATCH] Collected: {successes} OK, {failures} failed")
    return results


# ── Retry failed windows ──────────────────────────────────────────────────────

def retry_failed(match_dir: str, state: dict, step: str,
                 request_builder_fn) -> dict:
    """
    Find windows where step=failed, rebuild their requests, resubmit.
    request_builder_fn(window_id) -> request dict or None if not buildable.
    """
    failed_windows = [
        wid for wid, steps in state.get("windows", {}).items()
        if steps.get(step) == "failed"
    ]
    if not failed_windows:
        print(f"  [RETRY] No failed windows for {step}")
        return {}

    print(f"  [RETRY] Retrying {len(failed_windows)} failed windows for {step}")
    requests = []
    for wid in failed_windows:
        req = request_builder_fn(wid)
        if req:
            requests.append(req)

    if not requests:
        print(f"  [RETRY] Could not build requests for any failed windows")
        return {}

    batch_id = submit_batch(match_dir, state, requests, f"{step}_retry")
    batch    = poll_batch(batch_id)
    return collect_results(batch_id, match_dir, state, step)


if __name__ == "__main__":
    print("batch_runner.py — import this module to use in pipeline_runner.py")
    print("Functions: submit_batch, poll_batch, collect_results, retry_failed, with_retry")
