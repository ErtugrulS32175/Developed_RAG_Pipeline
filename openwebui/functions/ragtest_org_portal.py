"""
title: RAGTest Organization Portal
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
import time
import urllib.error
import urllib.request


PORTAL_PATH = "/ragtest-org"
API_URL = os.getenv(
    "RAGTEST_ORG_API_URL", "http://host.docker.internal:8000").rstrip("/")
GATEWAY_KEY = os.getenv("OPENWEBUI_GATEWAY_KEY", "")
JWT_SECRET = os.getenv("FORWARD_USER_INFO_HEADER_JWT_SECRET", "")


def _part(value):
    raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def signed_identity(user, *, now=None):
    """Create exactly the claim set the pinned OpenWebUI connection forwards."""
    if len(GATEWAY_KEY.encode("utf-8")) < 32 or len(JWT_SECRET.encode("utf-8")) < 32:
        raise RuntimeError("RAGTest organization bridge secrets are not configured")
    issued = int(time.time()) if now is None else now
    claims = {
        "sub": str(user.id),
        "email": str(user.email or ""),
        "name": str(user.name or ""),
        "role": str(user.role or "user"),
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


def _request(user, method, path, payload=None):
    body = None
    headers = {
        "Authorization": f"Bearer {GATEWAY_KEY}",
        "X-OpenWebUI-User-Jwt": signed_identity(user),
        "Accept": "application/json",
    }
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        API_URL + path, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.length is not None and response.length > 2_000_000:
                raise RuntimeError("organization response is too large")
            raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise RuntimeError("organization response is too large")
            return response.status, json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as error:
        raw = error.read(16_384)
        try:
            detail = json.loads(raw.decode("utf-8"))
        except Exception:
            detail = {"detail": "organization request refused"}
        return error.code, detail


async def _proxy(user, method, path, payload=None):
    return await asyncio.to_thread(_request, user, method, path, payload)


PORTAL_HTML = """<!doctype html>
<html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Organizasyon Mimarisi</title>
<style>
body{font:15px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;background:#f7f7f8;color:#171717}
section{background:white;border:1px solid #ddd;border-radius:12px;padding:20px;margin:16px 0}button{padding:9px 14px}
textarea{width:100%;min-height:420px;font:13px ui-monospace;box-sizing:border-box}li{margin:8px 0}.muted{color:#666}
</style>
<h1>Organizasyon Mimarisi</h1><p class="muted">Yetkiler OpenWebUI profilinden değil, RAGTest organizasyon sözleşmesinden gelir.</p>
<section><h2>Hesabım</h2><div id="me">Yükleniyor…</div><div id="subject" class="muted"></div></section>
<section><h2>Görebildiğim kullanıcılar</h2><button id="visible">Yönetim göreviyle getir</button><ul id="people"></ul></section>
<section id="admin" hidden><h2>Mimari düzenleyici</h2><p>Değişiklik bütün topolojiyi atomik olarak değiştirir; eski sürümle kayıt 409 döner.</p>
<textarea id="topology"></textarea><p><button id="save">Kaydet</button> <span id="result"></span></p></section>
<script>
const j=async(url,opt={})=>{const r=await fetch(url,{...opt,headers:{'Content-Type':'application/json',...(opt.headers||{})}});const b=await r.json();if(!r.ok)throw Error(b.detail||'İstek reddedildi');return b};
const text=(id,v)=>document.getElementById(id).textContent=v;
async function load(){try{const own=await j('/ragtest-org/api/identity');text('subject',`OpenWebUI kimliği: ${own.openwebui_subject}`)}catch(e){text('subject',e.message)}try{const me=await j('/ragtest-org/api/me');const m=me.membership;text('me',m?`${m.display_label} — Seviye ${m.level}, ${m.title} (${m.kind})`:'İş hesabı yok; yalnız mimari yönetim yetkisi var.');if(me.architecture_admin){document.getElementById('admin').hidden=false;document.getElementById('topology').value=JSON.stringify(await j('/ragtest-org/api/topology'),null,2)}}catch(e){text('me',e.message)}}
document.getElementById('visible').onclick=async()=>{const ul=document.getElementById('people');ul.replaceChildren();try{const b=await j('/ragtest-org/api/visible?reason_code=management_duty');for(const p of b.members){const li=document.createElement('li');li.textContent=`${p.display_label} — Seviye ${p.level}, ${p.title}`;ul.append(li)}}catch(e){const li=document.createElement('li');li.textContent=e.message;ul.append(li)}};
document.getElementById('save').onclick=async()=>{try{const b=JSON.parse(document.getElementById('topology').value);const payload={expected_version:b.architecture_version,name:b.name,positions:b.positions,members:b.members};const v=await j('/ragtest-org/api/topology',{method:'PUT',body:JSON.stringify(payload)});text('result',`Kaydedildi: sürüm ${v.architecture_version}`);await load()}catch(e){text('result',e.message)}};load();
</script></html>"""


class Event:
    async def event(self, event, __event_name__=None, __app__=None,
                    __id__=None, **_kwargs):
        about_me = (event.get("subject") or {}).get("id") == __id__
        if (__event_name__ == "system.startup.completed"
                or (__event_name__ == "function.enable_started" and about_me)):
            self._register(__app__)

    def _register(self, app):
        from fastapi import Body, Depends, Query
        from fastapi.responses import HTMLResponse, JSONResponse
        from open_webui.utils.auth import get_verified_user

        if any(getattr(route, "path", None) == PORTAL_PATH
               for route in app.routes):
            return

        async def page(_user=Depends(get_verified_user)):
            return HTMLResponse(PORTAL_HTML)

        async def me(user=Depends(get_verified_user)):
            status, body = await _proxy(user, "GET", "/v1/org/me")
            return JSONResponse(body, status_code=status)

        async def identity_info(user=Depends(get_verified_user)):
            return JSONResponse({"openwebui_subject": str(user.id)})

        async def visible(
                reason_code: str = Query(...),
                user=Depends(get_verified_user)):
            if reason_code not in {"management_duty", "security_review"}:
                return JSONResponse({"detail": "gecersiz neden"}, status_code=422)
            status, body = await _proxy(
                user, "GET", "/v1/org/visible-members?reason_code=" +
                reason_code)
            return JSONResponse(body, status_code=status)

        async def topology(user=Depends(get_verified_user)):
            status, body = await _proxy(
                user, "GET", "/v1/org/admin/topology")
            return JSONResponse(body, status_code=status)

        async def topology_put(
                payload: dict = Body(...), user=Depends(get_verified_user)):
            status, body = await _proxy(
                user, "PUT", "/v1/org/admin/topology", payload)
            return JSONResponse(body, status_code=status)

        app.add_api_route(PORTAL_PATH, page, methods=["GET"],
                          include_in_schema=False)
        app.add_api_route(PORTAL_PATH + "/api/me", me, methods=["GET"],
                          include_in_schema=False)
        app.add_api_route(PORTAL_PATH + "/api/identity", identity_info,
                          methods=["GET"], include_in_schema=False)
        app.add_api_route(PORTAL_PATH + "/api/visible", visible,
                          methods=["GET"], include_in_schema=False)
        app.add_api_route(PORTAL_PATH + "/api/topology", topology,
                          methods=["GET"], include_in_schema=False)
        app.add_api_route(PORTAL_PATH + "/api/topology", topology_put,
                          methods=["PUT"], include_in_schema=False)
