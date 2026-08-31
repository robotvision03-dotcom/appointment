#!/usr/bin/env python3
"""POST a mock Twilio incoming-call webhook and print the TwiML."""

from __future__ import annotations

import argparse
import sys

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description="Mock Twilio voice webhook")
    parser.add_argument("--url", default="http://127.0.0.1:38471/voice/incoming")
    parser.add_argument("--from-number", default="+989121111111")
    parser.add_argument("--to-number", default="+982100000000")
    parser.add_argument("--call-sid", default="CA_TEST_MOCK_001")
    args = parser.parse_args()

    payload = {
        "CallSid": args.call_sid,
        "AccountSid": "ACXXXXXXXX",
        "From": args.from_number,
        "To": args.to_number,
        "CallStatus": "ringing",
        "Direction": "inbound",
        "ApiVersion": "2010-04-01",
    }
    resp = httpx.post(args.url, data=payload, timeout=10.0)
    print(f"HTTP {resp.status_code}")
    print(resp.text)
    if resp.status_code != 200:
        return 1
    if "<Stream" not in resp.text and "<stream" not in resp.text.lower():
        print("Expected TwiML <Stream> element missing", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
