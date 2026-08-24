"""
title: Yanıtı Değerlendir
author: RAGTest
version: 1.0.0
required_open_webui_version: 0.11.0
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import re
import time
import urllib.error
import urllib.request

from pydantic import BaseModel


API_URL = os.getenv(
    "RAGTEST_ORG_API_URL", "http://host.docker.internal:8000").rstrip("/")
GATEWAY_KEY = os.getenv("OPENWEBUI_GATEWAY_KEY", "")
JWT_SECRET = os.getenv("FORWARD_USER_INFO_HEADER_JWT_SECRET", "")
_REF = re.compile(r"^[A-Za-z0-9_-]{43}$")
_MAX_RESPONSE = 16_384
_EVENT_CALL_TIMEOUT_SECONDS = 300
_REASONS = {
    "1": "incorrect",
    "2": "missing_evidence",
    "3": "outdated",
    "4": "unsafe",
    "5": "other",
}


def _part(value):
    raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _user_value(user, name, default=""):
    if isinstance(user, dict):
        return user.get(name, default)
    return getattr(user, name, default)


def signed_identity(user, *, now=None):
    if len(GATEWAY_KEY.encode("utf-8")) < 32 or len(
            JWT_SECRET.encode("utf-8")) < 32:
        raise RuntimeError("RAGTest feedback bridge secrets are not configured")
    issued = int(time.time()) if now is None else now
    claims = {
        "sub": str(_user_value(user, "id")),
        "email": str(_user_value(user, "email")),
        "name": str(_user_value(user, "name")),
        "role": str(_user_value(user, "role", "user") or "user"),
        "iss": "open-webui",
        "iat": issued,
        "exp": issued + 60,
    }
    head = _part({"alg": "HS256", "typ": "JWT"})
    body = _part(claims)
    signed = f"{head}.{body}".encode("ascii")
    tail = base64.urlsafe_b64encode(
        hmac.new(JWT_SECRET.encode("utf-8"), signed,
                 hashlib.sha256).digest()).rstrip(b"=").decode("ascii")
    return f"{head}.{body}.{tail}"


def feedback_reference(body):
    """Extract one marked locator from one unique clicked assistant message."""
    if (not isinstance(body, dict) or not isinstance(body.get("id"), str)
            or not isinstance(body.get("messages"), list)):
        return None
    clicked = [
        message for message in body["messages"]
        if isinstance(message, dict) and message.get("id") == body["id"]
    ]
    if len(clicked) != 1 or clicked[0].get("role") != "assistant":
        return None
    sources = clicked[0].get("sources")
    if not isinstance(sources, list):
        return None
    found = set()
    for source in sources:
        if not isinstance(source, dict):
            return None
        metadata = source.get("metadata")
        if not isinstance(metadata, list):
            return None
        for item in metadata:
            if not isinstance(item, dict):
                return None
            if "ragtest_feedback_ref" not in item:
                continue
            value = item["ragtest_feedback_ref"]
            if not isinstance(value, str) or _REF.fullmatch(value) is None:
                return None
            found.add(value)
    return next(iter(found)) if len(found) == 1 else None


def _request(user, payload):
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        API_URL + "/v1/reviews/feedback",
        data=raw,
        headers={
            "Authorization": f"Bearer {GATEWAY_KEY}",
            "X-OpenWebUI-User-Jwt": signed_identity(user),
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.length is not None and response.length > _MAX_RESPONSE:
                raise RuntimeError("feedback response is too large")
            body = response.read(_MAX_RESPONSE + 1)
            if len(body) > _MAX_RESPONSE:
                raise RuntimeError("feedback response is too large")
            return response.status, json.loads(body.decode("utf-8"))
    except urllib.error.HTTPError as error:
        error.read(_MAX_RESPONSE)
        return error.code, None


async def _event_call(caller, event):
    return await asyncio.wait_for(
        caller(event), timeout=_EVENT_CALL_TIMEOUT_SECONDS)


async def _notify(emitter, level, content):
    if emitter is not None:
        await emitter({
            "type": "notification",
            "data": {"type": level, "content": content},
        })


class Action:
    """Submit closed, content-free feedback for one checked RAG answer."""

    class Valves(BaseModel):
        priority: int = 0

    def __init__(self):
        self.valves = self.Valves()

    async def action(self, body: dict, __user__=None,
                     __event_emitter__=None, __event_call__=None):
        reference = feedback_reference(body)
        if reference is None or __event_call__ is None:
            await _notify(__event_emitter__, "warning",
                          "Bu mesaj geri bildirim için uygun değil.")
            return None
        try:
            choice = await _event_call(__event_call__, {
                "type": "input",
                "data": {
                    "title": "Yanıtı değerlendirin",
                    "message": "1 = Yararlı, 2 = Yararlı değil",
                    "placeholder": "1",
                },
            })
        except Exception:
            await _notify(__event_emitter__, "warning",
                          "Geri bildirim penceresi zaman aşımına uğradı.")
            return None
        selected = str(choice).strip()
        if selected not in {"1", "2"}:
            await _notify(__event_emitter__, "warning",
                          "Geçersiz geri bildirim seçimi.")
            return None
        verdict = "helpful" if selected == "1" else "not_helpful"
        reason = None
        if verdict == "not_helpful":
            try:
                reason_choice = await _event_call(__event_call__, {
                    "type": "input",
                    "data": {
                        "title": "Nedeni seçin",
                        "message": (
                            "1 = Yanlış, 2 = Kanıt eksik, 3 = Güncel değil, "
                            "4 = Güvensiz, 5 = Diğer"),
                        "placeholder": "1",
                    },
                })
            except Exception:
                await _notify(__event_emitter__, "warning",
                              "Neden seçimi zaman aşımına uğradı.")
                return None
            reason = _REASONS.get(str(reason_choice).strip())
            if reason is None:
                await _notify(__event_emitter__, "warning",
                              "Geçersiz neden seçimi.")
                return None
        payload = {
            "feedback_ref": reference,
            "verdict": verdict,
            "reason_code": reason,
        }
        try:
            status, response = await asyncio.to_thread(
                _request, __user__, payload)
            if (status != 200 or not isinstance(response, dict)
                    or frozenset(response) != frozenset({
                        "status", "revision", "review_open"})
                    or response.get("status") != "recorded"
                    or isinstance(response.get("revision"), bool)
                    or not isinstance(response.get("revision"), int)
                    or response["revision"] < 1
                    or type(response.get("review_open")) is not bool):
                raise RuntimeError("feedback response refused")
        except Exception:
            await _notify(__event_emitter__, "error",
                          "Geri bildirim kaydedilemedi.")
            return None
        await _notify(__event_emitter__, "success",
                      "Geri bildiriminiz kaydedildi.")
        return None
