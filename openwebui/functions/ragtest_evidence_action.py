"""
title: Kanıtı Göster
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
_TICKET = re.compile(r"^[A-Za-z0-9_-]{43}$")
_MAX_RESPONSE = 1_000_000
_EVENT_CALL_TIMEOUT_SECONDS = 300


def _part(value):
    raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _user_value(user, name, default=""):
    if isinstance(user, dict):
        return user.get(name, default)
    return getattr(user, name, default)


def signed_identity(user, *, now=None):
    """Use the same short-lived identity assertion as the organization bridge."""
    if len(GATEWAY_KEY.encode("utf-8")) < 32 or len(
            JWT_SECRET.encode("utf-8")) < 32:
        raise RuntimeError("RAGTest evidence bridge secrets are not configured")
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


def _request(user, path, payload):
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        API_URL + path,
        data=body,
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
                raise RuntimeError("evidence response is too large")
            raw = response.read(_MAX_RESPONSE + 1)
            if len(raw) > _MAX_RESPONSE:
                raise RuntimeError("evidence response is too large")
            return response.status, json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as error:
        error.read(16_384)
        return error.code, None


async def _proxy(user, path, payload):
    return await asyncio.to_thread(_request, user, path, payload)


async def _event_call(caller, event):
    """Bound an abandoned browser modal instead of holding a worker forever."""
    return await asyncio.wait_for(
        caller(event), timeout=_EVENT_CALL_TIMEOUT_SECONDS)


def _clicked_message(body):
    if not isinstance(body, dict) or not isinstance(body.get("id"), str):
        return None
    for message in body.get("messages") or []:
        if isinstance(message, dict) and message.get("id") == body["id"]:
            return message
    return None


def evidence_references(body):
    """Read only references persisted by official source events on this message."""
    message = _clicked_message(body)
    if message is None:
        return ()
    found = []
    for item in message.get("sources") or []:
        source = item.get("source") if isinstance(item, dict) else None
        value = source.get("id") if isinstance(source, dict) else None
        if (isinstance(value, str) and _REF.fullmatch(value)
                and value not in found):
            found.append(value)
    return tuple(found)


async def _notify(emitter, level, content):
    if emitter is not None:
        await emitter({
            "type": "notification",
            "data": {"type": level, "content": content},
        })


def _preview(value):
    if not isinstance(value, dict) or frozenset(value) != frozenset({
            "document_name", "page", "content_type", "passage"}):
        raise RuntimeError("evidence preview contract refused")
    name = value.get("document_name")
    page = value.get("page")
    passage = value.get("passage")
    if (not isinstance(name, str) or not name.strip() or len(name) > 500
            or isinstance(page, bool) or not isinstance(page, int) or page < 1
            or value.get("content_type") != "passage"
            or not isinstance(passage, str) or not passage
            or len(passage.encode("utf-8")) > 500_000):
        raise RuntimeError("evidence preview contract refused")
    return name, page, passage


class Action:
    """Fetch one freshly-authorized passage for a persisted citation."""

    class Valves(BaseModel):
        priority: int = 0

    def __init__(self):
        self.valves = self.Valves()

    async def action(self, body: dict, __user__=None,
                     __event_emitter__=None, __event_call__=None):
        references = evidence_references(body)
        if not references:
            await _notify(__event_emitter__, "warning",
                          "Bu mesajda görüntülenebilir kanıt yok.")
            return None
        if __event_call__ is None:
            await _notify(__event_emitter__, "warning",
                          "Kanıt önizleme penceresi açılamadı.")
            return None
        selected = 0
        if len(references) > 1:
            try:
                answer = await _event_call(__event_call__, {
                    "type": "input",
                    "data": {
                        "title": "Kanıt seçin",
                        "message": (
                            f"1 ile {len(references)} arasında kaynak "
                            "numarası girin."),
                        "placeholder": "1",
                    },
                })
            except Exception:
                await _notify(__event_emitter__, "warning",
                              "Kanıt seçimi zaman aşımına uğradı.")
                return None
            try:
                selected = int(str(answer).strip()) - 1
            except (TypeError, ValueError):
                selected = -1
            if selected < 0 or selected >= len(references):
                await _notify(__event_emitter__, "warning",
                              "Geçersiz kanıt seçimi.")
                return None
        try:
            status, ticket_body = await _proxy(
                __user__, "/v1/evidence/tickets",
                {"evidence_ref": references[selected]})
            if (status != 200 or not isinstance(ticket_body, dict)
                    or frozenset(ticket_body) != frozenset({
                        "ticket", "expires_in"})):
                raise RuntimeError("ticket refused")
            ticket = ticket_body.get("ticket")
            expires = ticket_body.get("expires_in")
            if (not isinstance(ticket, str) or _TICKET.fullmatch(ticket) is None
                    or isinstance(expires, bool) or not isinstance(expires, int)
                    or not 1 <= expires <= 60):
                raise RuntimeError("ticket refused")
            status, preview = await _proxy(
                __user__, "/v1/evidence/preview", {"ticket": ticket})
            if status != 200:
                raise RuntimeError("preview refused")
            name, page, passage = _preview(preview)
        except Exception:
            await _notify(__event_emitter__, "error",
                          "Kanıt önizlemesi yetkilendirilemedi.")
            return None
        # Interactive call data is displayed only in the active browser.  An
        # inline HTMLResponse would be persisted as a chat embed by OpenWebUI,
        # silently turning a freshly-authorized preview into durable content.
        try:
            await _event_call(__event_call__, {
                "type": "confirmation",
                "data": {
                    "title": f"{name} — Sayfa {page}",
                    "message": passage,
                },
            })
        except Exception:
            await _notify(__event_emitter__, "warning",
                          "Kanıt penceresi zaman aşımına uğradı.")
        return None
