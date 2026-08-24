"""
title: RAGTest Evaluation Dataset Portal
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

from fastapi import Request


PORTAL_PATH = "/ragtest-eval"
API_URL = os.getenv(
    "RAGTEST_EVAL_API_URL", "http://host.docker.internal:8000").rstrip("/")
GATEWAY_KEY = os.getenv("OPENWEBUI_GATEWAY_KEY", "")
JWT_SECRET = os.getenv("FORWARD_USER_INFO_HEADER_JWT_SECRET", "")
MAX_RESPONSE_BYTES = 1_000_000
MAX_IMPORT_BYTES = 16 * 1024 * 1024
MAX_IMPORT_CASES = 500
CASE_TYPES = frozenset({"metin", "sayisal", "tablo"})
DATASET_STATES = frozenset({"active", "retired"})
VERSION_STATES = frozenset({"draft", "published"})
_UUID = re.compile(
    r"\A[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z")
_SLUG = re.compile(r"\A[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_CASE_UUID = re.compile(
    r"\A[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z")


def _part(value):
    raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def signed_identity(user, *, now=None):
    """Create exactly the claim set the pinned OpenWebUI bridge forwards."""
    if len(GATEWAY_KEY.encode("utf-8")) < 32 or len(JWT_SECRET.encode("utf-8")) < 32:
        raise RuntimeError("RAGTest evaluation bridge secrets are not configured")
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
            if (response.length is not None
                    and response.length > MAX_RESPONSE_BYTES):
                raise RuntimeError("evaluation response is too large")
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise RuntimeError("evaluation response is too large")
            return response.status, json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as error:
        error.read(16_384)
        return error.code, {"detail": "evaluation request refused"}


async def _proxy(user, method, path, payload=None):
    return await asyncio.to_thread(_request, user, method, path, payload)


def _exact_text(value, *, minimum=1, maximum):
    return (type(value) is str and minimum <= len(value) <= maximum
            and "\x00" not in value)


def _identity(value):
    return type(value) is str and _UUID.fullmatch(value) is not None


def _case_text(value, maximum):
    if type(value) is not str or value != value.strip():
        return False
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        return False
    return (0 < len(encoded) <= maximum
            and not any(ord(char) < 0x20 for char in value))


def _dataset_create(payload):
    if type(payload) is not dict or set(payload) != {"slug", "label"}:
        return None
    if (not _exact_text(payload["slug"], maximum=80)
            or _SLUG.fullmatch(payload["slug"]) is None
            or not _exact_text(payload["label"], maximum=160)):
        return None
    return {"slug": payload["slug"], "label": payload["label"]}


def _draft_create(payload):
    if (type(payload) is not dict
            or set(payload) != {"expected_revision"}
            or type(payload["expected_revision"]) is not int
            or payload["expected_revision"] < 1):
        return None
    return {"expected_revision": payload["expected_revision"]}


def _import_cases(payload):
    if type(payload) is not dict or set(payload) != {"expected_revision", "cases"}:
        return None
    revision = payload["expected_revision"]
    cases = payload["cases"]
    if (type(revision) is not int or revision < 1 or type(cases) is not list
            or not 1 <= len(cases) <= MAX_IMPORT_CASES):
        return None
    clean = []
    for case in cases:
        if (type(case) is not dict
                or set(case) != {"case_key", "q", "key", "answer", "pages", "type"}
                or type(case["case_key"]) is not str
                or _CASE_UUID.fullmatch(case["case_key"]) is None
                or not _case_text(case["q"], 4_096)
                or not _case_text(case["key"], 16_384)
                or not _case_text(case["answer"], 16_384)
                or case["type"] not in CASE_TYPES
                or type(case["pages"]) is not list
                or not case["pages"]
                or any(type(page) is not int or page < 1
                       or page > 2_147_483_647
                       for page in case["pages"])
                or case["pages"] != sorted(set(case["pages"]))):
            return None
        clean.append({name: case[name] for name in (
            "case_key", "q", "key", "answer", "pages", "type")})
    case_keys = [case["case_key"] for case in clean]
    if case_keys != sorted(case_keys) or len(set(case_keys)) != len(case_keys):
        return None
    encoded = json.dumps(
        {"expected_revision": revision, "cases": clean},
        separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_IMPORT_BYTES:
        return None
    return {"expected_revision": revision, "cases": clean}


class _ImportTooLarge(ValueError):
    pass


def _closed_json_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate")
        value[key] = item
    return value


def _reject_json_constant(_value):
    raise ValueError("constant")


async def _read_import_request(request):
    """Bound the OpenWebUI-facing body before JSON allocates its tree."""
    length = request.headers.get("content-length")
    if length is not None:
        offered = int(length)
        if offered < 0:
            raise ValueError("length")
        if offered > MAX_IMPORT_BYTES:
            raise _ImportTooLarge("length")
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > MAX_IMPORT_BYTES:
            raise _ImportTooLarge("stream")
    text = bytes(raw).decode("utf-8", errors="strict")
    if not text or text.startswith("\ufeff"):
        raise ValueError("encoding")
    payload = json.loads(
        text, object_pairs_hook=_closed_json_pairs,
        parse_constant=_reject_json_constant)
    clean = _import_cases(payload)
    if clean is None:
        raise ValueError("shape")
    return clean


def _publish(payload):
    if (type(payload) is not dict
            or set(payload) != {"expected_revision", "expected_policy_epoch",
                                "expected_draft_sha256"}
            or any(type(payload[name]) is not int or payload[name] < 1
                   for name in ("expected_revision", "expected_policy_epoch"))
            or type(payload["expected_draft_sha256"]) is not str
            or _SHA256.fullmatch(payload["expected_draft_sha256"]) is None):
        return None
    return {name: payload[name] for name in (
        "expected_revision", "expected_policy_epoch", "expected_draft_sha256")}


def _retire(payload):
    if (type(payload) is not dict
            or set(payload) != {"expected_revision", "expected_policy_epoch"}
            or any(type(payload[name]) is not int or payload[name] < 1
                   for name in ("expected_revision", "expected_policy_epoch"))):
        return None
    return {name: payload[name] for name in (
        "expected_revision", "expected_policy_epoch")}


_DATASET_FIELDS = frozenset({
    "dataset_id", "slug", "label", "state", "revision", "owner_label",
    "policy_epoch", "current_version_id", "current_version_number", "versions",
})
_VERSION_FIELDS = frozenset({
    "version_id", "version_number", "state", "revision", "case_count",
    "content_sha256", "sealed_at",
})


def _closed_version(value):
    if type(value) is not dict or not set(value) <= _VERSION_FIELDS:
        return False
    if not {"version_id", "version_number", "state", "revision", "case_count"} <= set(value):
        return False
    if (not _identity(value["version_id"])
            or value["state"] not in VERSION_STATES
            or any(type(value[name]) is not int or value[name] < 1
                   for name in ("version_number", "revision"))
            or type(value["case_count"]) is not int
            or value["case_count"] < 0):
        return False
    if ("content_sha256" in value and value["content_sha256"] is not None
            and (type(value["content_sha256"]) is not str
                 or _SHA256.fullmatch(value["content_sha256"]) is None)):
        return False
    if ("sealed_at" in value and value["sealed_at"] is not None
            and not _exact_text(value["sealed_at"], maximum=64)):
        return False
    return True


def _closed_dataset(value):
    if type(value) is not dict or not set(value) <= _DATASET_FIELDS:
        return False
    if not {"dataset_id", "slug", "label", "state", "revision"} <= set(value):
        return False
    if (not _identity(value["dataset_id"])
            or not _exact_text(value["slug"], maximum=80)
            or _SLUG.fullmatch(value["slug"]) is None
            or not _exact_text(value["label"], maximum=160)
            or value["state"] not in DATASET_STATES
            or type(value["revision"]) is not int
            or value["revision"] < 1):
        return False
    if ("owner_label" in value
            and not _exact_text(value["owner_label"], minimum=0, maximum=160)):
        return False
    if ("policy_epoch" in value
            and (type(value["policy_epoch"]) is not int
                 or value["policy_epoch"] < 1)):
        return False
    if ("current_version_id" in value
            and value["current_version_id"] is not None
            and not _identity(value["current_version_id"])):
        return False
    if ("current_version_number" in value
            and value["current_version_number"] is not None
            and (type(value["current_version_number"]) is not int
                 or value["current_version_number"] < 1)):
        return False
    versions = value.get("versions", [])
    return (type(versions) is list and len(versions) <= 100
            and all(_closed_version(version) for version in versions))


def _closed_metadata(body):
    """Admit metadata only; case or run content has no response vocabulary."""
    if type(body) is not dict or set(body) != {"datasets"}:
        return False
    datasets = body["datasets"]
    return (type(datasets) is list and len(datasets) <= 100
            and all(_closed_dataset(dataset) for dataset in datasets))


def _closed_dataset_result(body):
    return (type(body) is dict and set(body) == {"dataset"}
            and _closed_dataset(body["dataset"]))


def _closed_version_result(body):
    return (type(body) is dict and set(body) == {"version"}
            and _closed_version(body["version"]))


def _closed_versions(body):
    return (type(body) is dict and set(body) == {"versions"}
            and type(body["versions"]) is list
            and len(body["versions"]) <= 100
            and all(_closed_version(version) for version in body["versions"]))


def _closed_import_result(body):
    return (_closed_version_result(body)
            and type(body["version"].get("content_sha256")) is str
            and _SHA256.fullmatch(
                body["version"]["content_sha256"]) is not None)


PORTAL_HTML = """<!doctype html>
<html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAGTest Değerlendirme Setleri</title>
<style>
body{font:15px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;background:#f7f7f8;color:#171717}
section{background:white;border:1px solid #ddd;border-radius:12px;padding:20px;margin:16px 0}button,input,textarea{padding:9px;margin:4px}textarea{width:100%;min-height:220px;box-sizing:border-box;font:13px ui-monospace}li{margin:8px 0}.muted{color:#666}
</style>
<h1>Değerlendirme Setleri</h1>
<p class="muted">Yetki RAGTest organizasyon sözleşmesinden gelir. OpenWebUI admin rolü yerel bir geçiş hakkı değildir.</p>
<section><h2>Setler ve sürümler</h2><button id="refresh">Yenile</button><span id="list-result"></span><ul id="datasets"></ul></section>
<section><h2>Yeni set</h2><input id="slug" placeholder="sabit-slug" maxlength="80"><input id="label" placeholder="Görünen ad" maxlength="160"><button id="create">Oluştur</button><span id="create-result"></span></section>
<section><h2>Taslak aç</h2><input id="draft-dataset" placeholder="dataset UUID"><input id="draft-revision" type="number" min="1" placeholder="beklenen revision"><button id="draft">Taslak oluştur</button><span id="draft-result"></span></section>
<section><h2>Kapalı vaka listesi içe aktar</h2><p class="muted">1–500 vaka; her nesne yalnız case_key, q, key, answer, pages, type alanlarını taşır. Veri tarayıcı deposuna yazılmaz ve başarıdan sonra giriş temizlenir.</p>
<input id="import-dataset" placeholder="dataset UUID"><input id="import-version" placeholder="version UUID"><input id="import-revision" type="number" min="1" placeholder="beklenen revision"><textarea id="cases" maxlength="16777216" placeholder='[{"case_key":"UUID","q":"...","key":"...","answer":"...","pages":[1],"type":"metin"}]'></textarea><button id="import">İçe aktar</button><span id="import-result"></span></section>
<section><h2>Sürümü yayımla</h2><input id="publish-dataset" placeholder="dataset UUID"><input id="publish-version" placeholder="version UUID"><input id="publish-revision" type="number" min="1" placeholder="beklenen revision"><input id="policy-epoch" type="number" min="1" placeholder="beklenen policy epoch"><input id="draft-sha256" maxlength="64" placeholder="beklenen draft SHA-256"><button id="publish">Yayımla</button><span id="publish-result"></span></section>
<section><h2>Seti emekliye ayır</h2><input id="retire-dataset" placeholder="dataset UUID"><input id="retire-revision" type="number" min="1" placeholder="beklenen revision"><input id="retire-policy-epoch" type="number" min="1" placeholder="beklenen policy epoch"><button id="retire">Emekliye ayır</button><span id="retire-result"></span></section>
<script>
const DATASET_STATES=new Set(['active','retired']);const VERSION_STATES=new Set(['draft','published']);const CASE_TYPES=new Set(['metin','sayisal','tablo']);const CASE_UUID=/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;const bytes=s=>new TextEncoder().encode(s).length;const cleanText=(s,max)=>typeof s==='string'&&s===s.trim()&&bytes(s)>=1&&bytes(s)<=max&&!/[\u0000-\u001f]/.test(s);
const j=async(url,opt={})=>{const r=await fetch(url,{...opt,headers:{'Content-Type':'application/json',...(opt.headers||{})}});const b=await r.json();if(!r.ok)throw Error(typeof b.detail==='string'?b.detail:'İstek reddedildi');return b};
const text=(id,v)=>document.getElementById(id).textContent=String(v);const value=id=>document.getElementById(id).value;const integer=id=>Number(value(id));
const exact=(o,keys,required)=>o&&o.constructor===Object&&Object.keys(o).every(k=>keys.includes(k))&&required.every(k=>Object.hasOwn(o,k));
const versionOk=v=>exact(v,['version_id','version_number','state','revision','case_count','content_sha256','sealed_at'],['version_id','version_number','state','revision','case_count'])&&VERSION_STATES.has(v.state);
const datasetOk=d=>exact(d,['dataset_id','slug','label','state','revision','owner_label','policy_epoch','current_version_id','current_version_number','versions'],['dataset_id','slug','label','state','revision'])&&DATASET_STATES.has(d.state)&&(!d.versions||(Array.isArray(d.versions)&&d.versions.every(versionOk)));
const caseOk=c=>exact(c,['case_key','q','key','answer','pages','type'],['case_key','q','key','answer','pages','type'])&&CASE_UUID.test(c.case_key)&&cleanText(c.q,4096)&&cleanText(c.key,16384)&&cleanText(c.answer,16384)&&CASE_TYPES.has(c.type)&&Array.isArray(c.pages)&&c.pages.length>=1&&c.pages.every(p=>Number.isInteger(p)&&p>=1&&p<=2147483647)&&c.pages.every((p,i)=>i===0||p>c.pages[i-1]);
const fillDataset=d=>{document.getElementById('draft-dataset').value=d.dataset_id;document.getElementById('draft-revision').value=d.revision;document.getElementById('retire-dataset').value=d.dataset_id;document.getElementById('retire-revision').value=d.revision;if(d.policy_epoch){document.getElementById('policy-epoch').value=d.policy_epoch;document.getElementById('retire-policy-epoch').value=d.policy_epoch}};
const fillVersion=(d,v)=>{fillDataset(d);document.getElementById('import-dataset').value=d.dataset_id;document.getElementById('import-version').value=v.version_id;document.getElementById('import-revision').value=v.revision;document.getElementById('publish-dataset').value=d.dataset_id;document.getElementById('publish-version').value=v.version_id;document.getElementById('publish-revision').value=v.revision;if(v.content_sha256)document.getElementById('draft-sha256').value=v.content_sha256};
const versionRow=(d,v)=>{const row=document.createElement('li');const label=document.createElement('span');label.textContent=`Sürüm ${v.version_number} — ${v.state}, ${v.case_count} vaka, revision ${v.revision}, id ${v.version_id}`;const use=document.createElement('button');use.textContent='Formlara aktar';use.onclick=()=>fillVersion(d,v);row.append(label,use);return row};
async function load(){const ul=document.getElementById('datasets');ul.replaceChildren();try{const b=await j('/ragtest-eval/api/datasets');if(!exact(b,['datasets'],['datasets'])||!Array.isArray(b.datasets)||!b.datasets.every(datasetOk))throw Error('Kapalı metadata sözleşmesi ihlal edildi');for(const d of b.datasets){const li=document.createElement('li');const title=document.createElement('strong');title.textContent=`${d.label} (${d.slug}) — ${d.state}, revision ${d.revision}, id ${d.dataset_id}, policy ${d.policy_epoch||'-'}`;const use=document.createElement('button');use.textContent='Set formlarına aktar';use.onclick=()=>fillDataset(d);const history=document.createElement('button');history.textContent='Sürüm geçmişi';const versions=document.createElement('ul');for(const v of d.versions||[])versions.append(versionRow(d,v));history.onclick=async()=>{try{const body=await j(`/ragtest-eval/api/datasets/${d.dataset_id}/versions`);if(!exact(body,['versions'],['versions'])||!Array.isArray(body.versions)||!body.versions.every(versionOk))throw Error('Kapalı sürüm sözleşmesi ihlal edildi');versions.replaceChildren();for(const v of body.versions)versions.append(versionRow(d,v))}catch(e){versions.replaceChildren();const error=document.createElement('li');error.textContent=e.message;versions.append(error)}};li.append(title,use,history,versions);ul.append(li)}text('list-result',`${b.datasets.length} set`)}catch(e){text('list-result',e.message)}}
document.getElementById('refresh').onclick=load;
document.getElementById('create').onclick=async()=>{try{await j('/ragtest-eval/api/datasets',{method:'POST',body:JSON.stringify({slug:value('slug'),label:value('label')})});text('create-result','Oluşturuldu');await load()}catch(e){text('create-result',e.message)}};
document.getElementById('draft').onclick=async()=>{try{await j(`/ragtest-eval/api/datasets/${value('draft-dataset')}/drafts`,{method:'POST',body:JSON.stringify({expected_revision:integer('draft-revision')})});text('draft-result','Taslak oluşturuldu');await load()}catch(e){text('draft-result',e.message)}};
document.getElementById('import').onclick=async()=>{try{const cases=JSON.parse(value('cases'));if(!Array.isArray(cases)||cases.length<1||cases.length>500||!cases.every(caseOk)||cases.some((c,i)=>i>0&&c.case_key<=cases[i-1].case_key))throw Error('Kapalı vaka sözleşmesi ihlal edildi');const body=JSON.stringify({expected_revision:integer('import-revision'),cases});if(bytes(body)>16777216)throw Error('Vaka aktarımı çok büyük');const result=await j(`/ragtest-eval/api/datasets/${value('import-dataset')}/versions/${value('import-version')}/cases/import`,{method:'POST',body});document.getElementById('draft-sha256').value=result.version.content_sha256;document.getElementById('publish-dataset').value=value('import-dataset');document.getElementById('publish-version').value=value('import-version');document.getElementById('publish-revision').value=result.version.revision;document.getElementById('cases').value='';text('import-result','İçe aktarıldı; yayın özeti bağlandı');await load()}catch(e){text('import-result',e.message)}};
document.getElementById('publish').onclick=async()=>{try{await j(`/ragtest-eval/api/datasets/${value('publish-dataset')}/versions/${value('publish-version')}/publish`,{method:'POST',body:JSON.stringify({expected_revision:integer('publish-revision'),expected_policy_epoch:integer('policy-epoch'),expected_draft_sha256:value('draft-sha256')})});text('publish-result','Yayımlandı');await load()}catch(e){text('publish-result',e.message)}};
document.getElementById('retire').onclick=async()=>{try{await j(`/ragtest-eval/api/datasets/${value('retire-dataset')}/retire`,{method:'POST',body:JSON.stringify({expected_revision:integer('retire-revision'),expected_policy_epoch:integer('retire-policy-epoch')})});text('retire-result','Emekliye ayrıldı');await load()}catch(e){text('retire-result',e.message)}};load();
</script></html>"""


class Event:
    async def event(self, event, __event_name__=None, __app__=None,
                    __id__=None, **_kwargs):
        about_me = (event.get("subject") or {}).get("id") == __id__
        if (__event_name__ == "system.startup.completed"
                or (__event_name__ == "function.enable_started" and about_me)):
            self._register(__app__)

    def _register(self, app):
        from fastapi import Body, Depends
        from fastapi.responses import HTMLResponse, JSONResponse
        from open_webui.utils.auth import get_verified_user

        if any(getattr(route, "path", None) == PORTAL_PATH
               for route in app.routes):
            return

        async def page(_user=Depends(get_verified_user)):
            return HTMLResponse(PORTAL_HTML)

        async def datasets(user=Depends(get_verified_user)):
            status, body = await _proxy(
                user, "GET", "/v1/eval/datasets?limit=100")
            if status < 400 and not _closed_metadata(body):
                return JSONResponse(
                    {"detail": "evaluation metadata contract refused"},
                    status_code=502)
            return JSONResponse(body, status_code=status)

        async def dataset_create(
                payload: dict = Body(...), user=Depends(get_verified_user)):
            clean = _dataset_create(payload)
            if clean is None:
                return JSONResponse(
                    {"detail": "invalid dataset request"}, status_code=422)
            status, body = await _proxy(
                user, "POST", "/v1/eval/datasets", clean)
            if status < 400 and not _closed_dataset_result(body):
                return JSONResponse(
                    {"detail": "evaluation dataset contract refused"},
                    status_code=502)
            return JSONResponse(body, status_code=status)

        async def versions(
                dataset_id: str, user=Depends(get_verified_user)):
            if not _identity(dataset_id):
                return JSONResponse(
                    {"detail": "invalid dataset id"}, status_code=422)
            status, body = await _proxy(
                user, "GET", "/v1/eval/datasets/" + dataset_id + "/versions")
            if status < 400 and not _closed_versions(body):
                return JSONResponse(
                    {"detail": "evaluation versions contract refused"},
                    status_code=502)
            return JSONResponse(body, status_code=status)

        async def draft_create(
                dataset_id: str, payload: dict = Body(...),
                user=Depends(get_verified_user)):
            clean = _draft_create(payload)
            if not _identity(dataset_id) or clean is None:
                return JSONResponse(
                    {"detail": "invalid draft request"}, status_code=422)
            status, body = await _proxy(
                user, "POST", "/v1/eval/datasets/" + dataset_id +
                "/drafts", clean)
            if status < 400 and not _closed_version_result(body):
                return JSONResponse(
                    {"detail": "evaluation version contract refused"},
                    status_code=502)
            return JSONResponse(body, status_code=status)

        async def cases_import(
                dataset_id: str, version_id: str, request: Request,
                user=Depends(get_verified_user)):
            if not _identity(dataset_id) or not _identity(version_id):
                return JSONResponse(
                    {"detail": "invalid case import"}, status_code=422)
            try:
                clean = await _read_import_request(request)
            except _ImportTooLarge:
                return JSONResponse(
                    {"detail": "case import too large"}, status_code=413)
            except (UnicodeDecodeError, json.JSONDecodeError, RecursionError,
                    ValueError):
                return JSONResponse(
                    {"detail": "invalid case import"}, status_code=422)
            status, body = await _proxy(
                user, "POST", "/v1/eval/datasets/" + dataset_id +
                "/versions/" + version_id + "/cases/import", clean)
            if status < 400 and not _closed_import_result(body):
                return JSONResponse(
                    {"detail": "evaluation version contract refused"},
                    status_code=502)
            return JSONResponse(body, status_code=status)

        async def publish(
                dataset_id: str, version_id: str, payload: dict = Body(...),
                user=Depends(get_verified_user)):
            clean = _publish(payload)
            if (not _identity(dataset_id) or not _identity(version_id)
                    or clean is None):
                return JSONResponse(
                    {"detail": "invalid publish request"}, status_code=422)
            status, body = await _proxy(
                user, "POST", "/v1/eval/datasets/" + dataset_id +
                "/versions/" + version_id + "/publish", clean)
            if status < 400 and not _closed_version_result(body):
                return JSONResponse(
                    {"detail": "evaluation version contract refused"},
                    status_code=502)
            return JSONResponse(body, status_code=status)

        async def retire(
                dataset_id: str, payload: dict = Body(...),
                user=Depends(get_verified_user)):
            clean = _retire(payload)
            if not _identity(dataset_id) or clean is None:
                return JSONResponse(
                    {"detail": "invalid retire request"}, status_code=422)
            status, body = await _proxy(
                user, "POST", "/v1/eval/datasets/" + dataset_id + "/retire",
                clean)
            if status < 400 and not _closed_dataset_result(body):
                return JSONResponse(
                    {"detail": "evaluation dataset contract refused"},
                    status_code=502)
            return JSONResponse(body, status_code=status)

        app.add_api_route(PORTAL_PATH, page, methods=["GET"],
                          include_in_schema=False)
        app.add_api_route(PORTAL_PATH + "/api/datasets", datasets,
                          methods=["GET"], include_in_schema=False)
        app.add_api_route(PORTAL_PATH + "/api/datasets", dataset_create,
                          methods=["POST"], include_in_schema=False)
        app.add_api_route(
            PORTAL_PATH + "/api/datasets/{dataset_id}/versions", versions,
            methods=["GET"], include_in_schema=False)
        app.add_api_route(
            PORTAL_PATH + "/api/datasets/{dataset_id}/drafts", draft_create,
            methods=["POST"], include_in_schema=False)
        app.add_api_route(
            PORTAL_PATH + "/api/datasets/{dataset_id}/versions/"
            "{version_id}/cases/import", cases_import, methods=["POST"],
            include_in_schema=False)
        app.add_api_route(
            PORTAL_PATH + "/api/datasets/{dataset_id}/versions/"
            "{version_id}/publish", publish, methods=["POST"],
            include_in_schema=False)
        app.add_api_route(
            PORTAL_PATH + "/api/datasets/{dataset_id}/retire", retire,
            methods=["POST"], include_in_schema=False)
