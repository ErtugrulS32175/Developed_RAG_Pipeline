"""
title: RAGTest Organization Portal
author: RAGTest
version: 2.0.0
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
import urllib.parse
import urllib.request
import uuid

from fastapi import Request


PORTAL_PATH = "/ragtest-org"
API_URL = os.getenv(
    "RAGTEST_ORG_API_URL", "http://host.docker.internal:8000").rstrip("/")
GATEWAY_KEY = os.getenv("OPENWEBUI_GATEWAY_KEY", "")
JWT_SECRET = os.getenv("FORWARD_USER_INFO_HEADER_JWT_SECRET", "")
MAX_RESPONSE_BYTES = 2_000_000
MUTATION_HEADER = "X-RAGTest-Portal"
MUTATION_VALUE = "same-origin"
EVENT_ACTIONS = frozenset({
    "monitor_view", "topology_read", "topology_change", "access_preview",
    "review_queue_view", "review_decision", "events_view",
    "membership_change", "retention_inventory_view",
    "retention_policy_change", "legal_hold_change", "purge_schedule",
    "purge_execute",
})
EVENT_DECISIONS = frozenset({"allowed", "denied"})
EVENT_REASONS = frozenset({
    "management_duty", "security_review", "system_operation",
    "policy_preview",
})


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
            if (response.length is not None
                    and response.length > MAX_RESPONSE_BYTES):
                raise RuntimeError("organization response is too large")
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise RuntimeError("organization response is too large")
            decoded = json.loads(raw.decode("utf-8"))
            if type(decoded) is not dict:
                raise RuntimeError("organization response shape is invalid")
            return response.status, decoded
    except urllib.error.HTTPError as error:
        raw = error.read(16_384)
        try:
            detail = json.loads(raw.decode("utf-8"))
        except Exception:
            detail = {"detail": "organization request refused"}
        return error.code, detail


async def _proxy(user, method, path, payload=None):
    return await asyncio.to_thread(_request, user, method, path, payload)


def _closed_payload(payload, *, required, optional=()):
    if type(payload) is not dict:
        return False
    keys = set(payload)
    return set(required) <= keys <= (set(required) | set(optional))


def _mutation_allowed(request):
    content_type = request.headers.get("content-type", "").split(";", 1)[0]
    return (request.headers.get(MUTATION_HEADER) == MUTATION_VALUE
            and content_type.strip().lower() == "application/json")


def _source_hash(source):
    digest = hashlib.sha256(source.encode("utf-8")).digest()
    return base64.b64encode(digest).decode("ascii")


PORTAL_STYLE = """
:root{color-scheme:light dark;--bg:#f4f6f8;--panel:#fff;--ink:#17202a;--muted:#61707d;--line:#d9e0e7;--brand:#126b5b;--danger:#a52929;--soft:#e8f4f1}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}header{background:#102f2a;color:#fff;padding:28px max(20px,calc((100vw - 1280px)/2))}header h1{margin:0 0 6px;font-size:28px}.shell{max-width:1280px;margin:auto;padding:20px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px;margin-bottom:16px;box-shadow:0 2px 10px #0000000b}.panel h2,.panel h3{margin-top:0}.muted{color:var(--muted)}.badge{display:inline-block;border-radius:999px;background:var(--soft);color:#0c5c4d;padding:3px 9px;font-size:12px;font-weight:700}.danger{color:var(--danger)}nav{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}button,.button{border:1px solid #9aa7b2;border-radius:8px;background:var(--panel);color:var(--ink);padding:8px 12px;cursor:pointer}button.primary{background:var(--brand);border-color:var(--brand);color:#fff}button.danger{border-color:#c85b5b}button:disabled{opacity:.5;cursor:not-allowed}input,select{width:100%;padding:8px;border:1px solid #aab5bf;border-radius:7px;background:var(--panel);color:var(--ink)}label{display:block;font-weight:650;margin:8px 0 4px}.form-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;align-items:end}.check{display:flex;align-items:center;gap:7px;font-weight:500}.check input{width:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;border-bottom:1px solid var(--line);padding:9px;vertical-align:top}.table-wrap{overflow:auto}.tree,.tree ul{list-style:none;padding-left:20px}.tree>li{padding-left:0}.tree li{margin:8px 0}.tree-node{border-left:4px solid var(--brand);background:var(--soft);border-radius:7px;padding:8px 10px}.notice{min-height:22px;margin-top:8px}.notice[data-kind=error]{color:var(--danger)}.tabs[hidden],section[hidden]{display:none}.mono{font:12px ui-monospace,monospace;overflow-wrap:anywhere}.toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.toolbar>*{width:auto}.metric{font-size:26px;font-weight:750}.empty{padding:18px;text-align:center;color:var(--muted)}@media(max-width:760px){.shell{padding:12px}th,td{min-width:130px}}
@media(prefers-color-scheme:dark){:root{--bg:#101614;--panel:#18211f;--ink:#ecf3f1;--muted:#a9b6b2;--line:#35443f;--soft:#173b34;--brand:#2f9b87}.badge{color:#b9fff0}}
""".strip()


PORTAL_SCRIPT = r"""
'use strict';
const state={me:null,topology:null,policy:null,documents:[],documentCursor:null,selected:null,eventCursor:null};
const by=id=>document.getElementById(id);
const text=(id,value)=>{by(id).textContent=value==null?'':String(value)};
const clear=id=>by(id).replaceChildren();
const notice=(id,value,kind='ok')=>{text(id,value);by(id).dataset.kind=kind};
const node=(tag,attrs={},children=[])=>{const n=document.createElement(tag);for(const [key,value] of Object.entries(attrs)){if(key==='class')n.className=value;else if(key==='text')n.textContent=value;else if(key==='disabled')n.disabled=Boolean(value);else n.setAttribute(key,String(value))}for(const child of children)n.append(child);return n};
const option=(value,label,selected=false)=>node('option',{value,text:label,...(selected?{selected:'selected'}:{})});
async function j(url,opt={}){const method=(opt.method||'GET').toUpperCase();const headers={'Accept':'application/json',...(opt.headers||{})};if(method!=='GET'){headers['Content-Type']='application/json';headers['X-RAGTest-Portal']='same-origin'}const controller=new AbortController();const timer=setTimeout(()=>controller.abort(),12000);try{const response=await fetch(url,{...opt,method,headers,credentials:'same-origin',signal:controller.signal});const raw=await response.text();let body={};try{body=raw?JSON.parse(raw):{}}catch(_error){throw Error('Sunucu kapalı biçimde yanıt vermedi')}if(!response.ok)throw Error(typeof body.detail==='string'?body.detail:'İstek reddedildi');return body}finally{clearTimeout(timer)}}
function showTab(name){for(const tab of document.querySelectorAll('[data-tab]'))tab.hidden=tab.dataset.tab!==name;for(const button of document.querySelectorAll('[data-tab-button]'))button.setAttribute('aria-pressed',String(button.dataset.tabButton===name))}
function selectControl(values,current){const s=node('select');for(const value of values)s.append(option(value,value,current===value));return s}
function positionControl(current,allowEmpty=false){const s=node('select');if(allowEmpty)s.append(option('','—',!current));for(const p of state.topology.positions)s.append(option(p.id,p.title,p.id===current));return s}
async function loadAccount(){const own=await j('/ragtest-org/api/identity');text('subject',own.openwebui_subject);const me=await j('/ragtest-org/api/me');state.me=me;const membership=me.membership;text('account-name',membership?membership.display_label:'Mimari yönetici');text('account-level',membership?`Seviye ${membership.level} · ${membership.title}`:'İçerik üyeliği yok');text('account-role',membership?membership.app_role:'content-blind');by('admin-nav').hidden=!me.architecture_admin;if(me.architecture_admin){await Promise.all([loadTopology(),loadPolicy(),loadDocuments(true)])}}
async function loadVisible(){clear('people');notice('visible-result','Yükleniyor…');try{const body=await j('/ragtest-org/api/visible?reason_code=management_duty');if(!body.members.length)by('people').append(node('li',{class:'empty',text:'Görünür alt hesap yok.'}));for(const p of body.members)by('people').append(node('li',{text:`${p.display_label} · Seviye ${p.level} · ${p.title}`}));notice('visible-result',`${body.members.length} hesap`)}catch(error){notice('visible-result',error.message,'error')}}
async function loadReviews(){clear('cases');notice('review-result','Yükleniyor…');try{const body=await j('/ragtest-org/api/reviews?reason_code=management_duty');if(!body.cases.length)by('cases').append(node('li',{class:'empty',text:'Bekleyen inceleme yok.'}));for(const item of body.cases){const li=node('li');li.append(node('span',{text:`${item.subject_label} · ${item.position_title} · ${item.trigger_code}`}));for(const [caption,decision,resolution] of [['Düzeltildi','resolved','corrected'],['Sorun yok','dismissed','no_issue']]){const button=node('button',{type:'button',text:caption});button.addEventListener('click',async()=>{try{await j(`/ragtest-org/api/reviews/${item.case_id}/decision`,{method:'POST',body:JSON.stringify({expected_revision:item.revision,expected_policy_epoch:item.policy_epoch,decision,resolution_code:resolution,reason_code:'management_duty'})});await loadReviews()}catch(error){notice('review-result',error.message,'error')}});li.append(' ',button)}by('cases').append(li)}notice('review-result',`${body.cases.length} vaka`)}catch(error){notice('review-result',error.message,'error')}}
function treeBranch(id,children){const p=state.topology.positions.find(item=>item.id===id);const flags=[p.kind,p.can_monitor_descendants?'alt katları izler':'izlemez',p.protected_from_monitoring?'korumalı':'izlenebilir'];const li=node('li');li.append(node('div',{class:'tree-node',text:`${p.title} · ${flags.join(' · ')}`}));const nested=children.get(id)||[];if(nested.length){const ul=node('ul');for(const child of nested.sort((a,b)=>a.title.localeCompare(b.title)))ul.append(treeBranch(child.id,children));li.append(ul)}return li}
function renderTree(){clear('org-tree');const children=new Map();for(const p of state.topology.positions){const key=p.parent_id||'root';if(!children.has(key))children.set(key,[]);children.get(key).push(p)}const roots=children.get('root')||[];for(const root of roots)by('org-tree').append(treeBranch(root.id,children))}
function renderPositions(){clear('position-rows');for(const p of state.topology.positions){const row=node('tr');const title=node('input',{value:p.title,maxlength:'200'});title.addEventListener('change',()=>p.title=title.value.trim());const kind=selectControl(['root','manager','member'],p.kind);kind.addEventListener('change',()=>p.kind=kind.value);const parent=positionControl(p.parent_id,true);parent.addEventListener('change',()=>p.parent_id=parent.value||null);const monitor=node('input',{type:'checkbox'});monitor.checked=p.can_monitor_descendants;monitor.addEventListener('change',()=>p.can_monitor_descendants=monitor.checked);const protect=node('input',{type:'checkbox'});protect.checked=p.protected_from_monitoring;protect.addEventListener('change',()=>p.protected_from_monitoring=protect.checked);const remove=node('button',{type:'button',class:'danger',text:'Kaldır'});remove.addEventListener('click',()=>{state.topology.positions=state.topology.positions.filter(item=>item.id!==p.id);renderTopology()});for(const child of [title,kind,parent,monitor,protect,remove])row.append(node('td',{},[child]));by('position-rows').append(row)}}
function memberPayload(member){return{issuer:'open-webui',subject:member.subject,position_id:member.position_id,display_label:member.display_label,app_role:member.app_role,state:member.state}}
function renderMembers(){clear('member-rows');for(const m of state.topology.members){const row=node('tr');const label=node('input',{value:m.display_label,maxlength:'200'});label.addEventListener('change',()=>m.display_label=label.value.trim());const subject=node('span',{class:'mono',text:m.subject});const position=positionControl(m.position_id);position.addEventListener('change',()=>m.position_id=position.value);const role=selectControl(['reader','editor','admin'],m.app_role);role.addEventListener('change',()=>m.app_role=role.value);const status=selectControl(['active','pending','suspended'],m.state);status.addEventListener('change',()=>m.state=status.value);const actions=node('div',{class:'toolbar'});if(m.identity_id){const apply=node('button',{type:'button',text:'Üyeliği uygula'});apply.addEventListener('click',()=>applyMember(m));actions.append(apply)}const remove=node('button',{type:'button',class:'danger',text:'Taslakta kaldır'});remove.addEventListener('click',()=>{state.topology.members=state.topology.members.filter(item=>item!==m);renderTopology()});actions.append(remove);for(const child of [label,subject,position,role,status,actions])row.append(node('td',{},[child]));by('member-rows').append(row)}}
function renderTopology(){text('architecture-version',state.topology.architecture_version);text('policy-epoch',state.topology.policy_epoch);by('org-name').value=state.topology.name;renderTree();renderPositions();renderMembers();for(const id of ['new-position-parent','new-member-position']){const select=by(id);select.replaceChildren();if(id==='new-position-parent')select.append(option('','— kök —'));for(const p of state.topology.positions)select.append(option(p.id,p.title))}}
async function loadTopology(){state.topology=await j('/ragtest-org/api/topology');renderTopology()}
async function saveTopology(){notice('topology-result','Kaydediliyor…');try{const payload={expected_version:state.topology.architecture_version,name:by('org-name').value.trim(),positions:state.topology.positions.map(({id,parent_id,title,kind,can_monitor_descendants,protected_from_monitoring})=>({id,parent_id,title,kind,can_monitor_descendants,protected_from_monitoring})),members:state.topology.members.map(memberPayload)};await j('/ragtest-org/api/topology',{method:'PUT',body:JSON.stringify(payload)});await loadTopology();notice('topology-result','Mimari atomik olarak kaydedildi.')}catch(error){notice('topology-result',error.message,'error')}}
async function applyMember(member){notice('member-result','Uygulanıyor…');try{await j(`/ragtest-org/api/members/${member.identity_id}`,{method:'PUT',body:JSON.stringify({expected_architecture_version:state.topology.architecture_version,expected_policy_epoch:state.topology.policy_epoch,state:member.state,app_role:member.app_role,position_id:member.position_id})});await loadTopology();notice('member-result','Üyelik güncellendi.')}catch(error){notice('member-result',error.message,'error')}}
function addPosition(event){event.preventDefault();const kind=by('new-position-kind').value;state.topology.positions.push({id:crypto.randomUUID(),parent_id:kind==='root'?null:(by('new-position-parent').value||null),title:by('new-position-title').value.trim(),kind,can_monitor_descendants:kind==='root'||by('new-position-monitor').checked,protected_from_monitoring:kind==='root'||by('new-position-protected').checked});event.target.reset();renderTopology()}
function addMember(event){event.preventDefault();state.topology.members.push({issuer:'open-webui',subject:by('new-member-subject').value.trim(),position_id:by('new-member-position').value,display_label:by('new-member-label').value.trim(),app_role:by('new-member-role').value,state:by('new-member-state').value});event.target.reset();renderTopology()}
async function loadEvents(reset=true){if(reset){state.eventCursor=null;clear('event-rows')}notice('event-result','Yükleniyor…');try{const query=new URLSearchParams({limit:'50'});for(const id of ['event-action','event-decision','event-reason'])if(by(id).value)query.set(id.replace('event-',''),by(id).value);if(state.eventCursor){query.set('before_created_at',state.eventCursor.before_created_at);query.set('before_id',state.eventCursor.before_id)}const body=await j('/ragtest-org/api/events?'+query.toString());for(const item of body.events){const row=node('tr');for(const value of [item.created_at,item.action,item.decision,item.reason_code,item.request_id])row.append(node('td',{text:value||'—'}));by('event-rows').append(row)}state.eventCursor=body.next_cursor;by('events-more').disabled=!body.has_more;notice('event-result',`${body.events.length} olay yüklendi.`)}catch(error){notice('event-result',error.message,'error')}}
async function loadPolicy(){state.policy=await j('/ragtest-org/api/retention-policy');by('retention-days').value=state.policy.archive_retention_days;text('retention-revision',state.policy.revision);text('retention-epoch',state.policy.policy_epoch)}
async function savePolicy(){notice('retention-result','Kaydediliyor…');try{await j('/ragtest-org/api/retention-policy',{method:'PUT',body:JSON.stringify({expected_revision:state.policy.revision,expected_policy_epoch:state.policy.policy_epoch,archive_retention_days:Number(by('retention-days').value)})});await Promise.all([loadPolicy(),loadTopology()]);notice('retention-result','Saklama politikası güncellendi.')}catch(error){notice('retention-result',error.message,'error')}}
function renderDocuments(){clear('document-rows');for(const item of state.documents){const row=node('tr');for(const value of [item.document_id,item.status,item.revision,item.active_hold_count,item.latest_purge_state||'—'])row.append(node('td',{class:value===item.document_id?'mono':'',text:value}));const manage=node('button',{type:'button',text:'Yönet'});manage.addEventListener('click',()=>selectDocument(item));row.append(node('td',{},[manage]));by('document-rows').append(row)}}
async function loadDocuments(reset=true){if(reset){state.documents=[];state.documentCursor=null}notice('document-result','Yükleniyor…');try{const query=new URLSearchParams({limit:'50'});if(state.documentCursor){query.set('before_uploaded_at',state.documentCursor.before_uploaded_at);query.set('before_id',state.documentCursor.before_id)}const body=await j('/ragtest-org/api/retention-documents?'+query.toString());state.documents.push(...body.documents);state.documentCursor=body.next_cursor;renderDocuments();by('documents-more').disabled=!body.has_more;notice('document-result',`${body.documents.length} belge yaşam döngüsü yüklendi.`)}catch(error){notice('document-result',error.message,'error')}}
async function selectDocument(item){state.selected=item;by('document-detail').hidden=false;text('selected-document',item.document_id);text('selected-status',`${item.status} · revizyon ${item.revision}`);await Promise.all([loadHolds(),loadPurgeJobs()])}
async function loadHolds(){clear('hold-rows');try{const body=await j(`/ragtest-org/api/documents/${state.selected.document_id}/holds`);for(const hold of body.holds){const row=node('tr');for(const value of [hold.reason_code,hold.state,hold.revision,hold.created_at])row.append(node('td',{text:value||'—'}));const action=node('td');if(hold.state==='active'){const release=node('button',{type:'button',class:'danger',text:'Hold kaldır'});release.addEventListener('click',()=>releaseHold(hold));action.append(release)}row.append(action);by('hold-rows').append(row)}}catch(error){notice('hold-result',error.message,'error')}}
async function createHold(){notice('hold-result','Oluşturuluyor…');try{await j(`/ragtest-org/api/documents/${state.selected.document_id}/holds`,{method:'POST',body:JSON.stringify({expected_document_revision:state.selected.revision,expected_policy_epoch:state.policy.policy_epoch,reason_code:by('hold-reason').value})});await refreshRetention();notice('hold-result','Legal hold oluşturuldu; bekleyen purge işleri iptal edildi.')}catch(error){notice('hold-result',error.message,'error')}}
async function releaseHold(hold){notice('hold-result','Kaldırılıyor…');try{await j(`/ragtest-org/api/documents/${state.selected.document_id}/holds/${hold.id}/release`,{method:'POST',body:JSON.stringify({expected_revision:hold.revision,expected_policy_epoch:state.policy.policy_epoch})});await refreshRetention();notice('hold-result','Legal hold kaldırıldı.')}catch(error){notice('hold-result',error.message,'error')}}
async function loadPurgeJobs(){clear('purge-rows');try{const body=await j(`/ragtest-org/api/documents/${state.selected.document_id}/purges`);for(const job of body.jobs){const row=node('tr');for(const value of [job.state,job.eligible_at,job.attempt_count,job.failure_code||'—'])row.append(node('td',{text:value||'—'}));by('purge-rows').append(row)}}catch(error){notice('purge-result',error.message,'error')}}
async function schedulePurge(){if(!confirm('Bu işlem retention süresi ve hold kapıları geçerse geri döndürülemez silmeyi kuyruğa alır. Devam edilsin mi?'))return;notice('purge-result','Planlanıyor…');try{await j(`/ragtest-org/api/documents/${state.selected.document_id}/purges`,{method:'POST',body:JSON.stringify({expected_document_revision:state.selected.revision,expected_policy_epoch:state.policy.policy_epoch})});await refreshRetention();notice('purge-result','Purge işi kuyruğa alındı.')}catch(error){notice('purge-result',error.message,'error')}}
async function refreshRetention(){const selectedId=state.selected&&state.selected.document_id;await Promise.all([loadPolicy(),loadDocuments(true)]);if(selectedId){const updated=state.documents.find(item=>item.document_id===selectedId);if(updated)await selectDocument(updated);else by('document-detail').hidden=true}}
async function start(){for(const button of document.querySelectorAll('[data-tab-button]'))button.addEventListener('click',()=>showTab(button.dataset.tabButton));by('visible-load').addEventListener('click',loadVisible);by('reviews-load').addEventListener('click',loadReviews);by('topology-save').addEventListener('click',saveTopology);by('position-form').addEventListener('submit',addPosition);by('member-form').addEventListener('submit',addMember);by('events-load').addEventListener('click',()=>loadEvents(true));by('events-more').addEventListener('click',()=>loadEvents(false));by('retention-save').addEventListener('click',savePolicy);by('documents-reload').addEventListener('click',()=>loadDocuments(true));by('documents-more').addEventListener('click',()=>loadDocuments(false));by('hold-create').addEventListener('click',createHold);by('purge-schedule').addEventListener('click',schedulePurge);try{await loadAccount();notice('startup-result','Yetkiler veritabanından doğrulandı.')}catch(error){notice('startup-result',error.message,'error')}}
start();
""".strip()


PORTAL_HTML = """<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RAGTest Organizasyon Yönetimi</title><style>__STYLE__</style></head>
<body><header><h1>RAGTest Organizasyon Yönetimi</h1><div>Kimlik ve yetkiler OpenWebUI profil rolünden değil, imzalı kimlik ve tenant sözleşmesinden gelir.</div></header><main class="shell">
<div id="startup-result" class="notice" role="status"></div>
<nav><button type="button" data-tab-button="account" aria-pressed="true">Hesabım</button><span id="admin-nav" class="tabs" hidden><button type="button" data-tab-button="topology">Mimari</button><button type="button" data-tab-button="events">Yönetişim</button><button type="button" data-tab-button="retention">Saklama ve silme</button></span></nav>
<section data-tab="account"><div class="grid"><article class="panel"><span class="badge">Doğrulanmış hesap</span><h2 id="account-name">Yükleniyor…</h2><div id="account-level"></div><div id="account-role" class="muted"></div><p class="mono">OpenWebUI subject: <span id="subject"></span></p></article><article class="panel"><h2>Görebildiğim kullanıcılar</h2><p class="muted">Yalnız yönetim kapsamındaki strict descendant hesaplar.</p><button id="visible-load" type="button">Hesapları getir</button><span id="visible-result" class="notice"></span><ul id="people"></ul></article></div><article class="panel"><h2>İnceleme kuyruğum</h2><p class="muted">İçerik gösterilmez; yalnız yetkili olduğunuz hesaplara ait kapalı vaka metadatası.</p><button id="reviews-load" type="button">Kuyruğu yenile</button><span id="review-result" class="notice"></span><ul id="cases"></ul></article></section>
<section data-tab="topology" hidden><article class="panel"><div class="toolbar"><span class="badge">Mimari sürüm <span id="architecture-version"></span></span><span class="badge">Politika epoch <span id="policy-epoch"></span></span></div><label for="org-name">Organizasyon adı</label><input id="org-name" maxlength="200"><h2>Hiyerarşi</h2><ul id="org-tree" class="tree"></ul></article>
<article class="panel"><h2>Pozisyonlar</h2><form id="position-form" class="form-grid"><div><label for="new-position-title">Başlık</label><input id="new-position-title" required maxlength="200"></div><div><label for="new-position-kind">Tür</label><select id="new-position-kind"><option value="manager">manager</option><option value="member">member</option><option value="root">root</option></select></div><div><label for="new-position-parent">Üst pozisyon</label><select id="new-position-parent"></select></div><label class="check"><input id="new-position-monitor" type="checkbox">Alt katları izler</label><label class="check"><input id="new-position-protected" type="checkbox">İzlemeden korumalı</label><button class="primary" type="submit">Taslağa ekle</button></form><div class="table-wrap"><table><thead><tr><th>Başlık</th><th>Tür</th><th>Üst</th><th>İzler</th><th>Korumalı</th><th></th></tr></thead><tbody id="position-rows"></tbody></table></div></article>
<article class="panel"><h2>Üyeler</h2><form id="member-form" class="form-grid"><div><label for="new-member-label">Görünen ad</label><input id="new-member-label" required maxlength="200"></div><div><label for="new-member-subject">OpenWebUI user id</label><input id="new-member-subject" required maxlength="200" pattern="[ -~]+"></div><div><label for="new-member-position">Pozisyon</label><select id="new-member-position"></select></div><div><label for="new-member-role">İçerik rolü</label><select id="new-member-role"><option>reader</option><option>editor</option><option>admin</option></select></div><div><label for="new-member-state">Durum</label><select id="new-member-state"><option>active</option><option>pending</option><option>suspended</option></select></div><button class="primary" type="submit">Taslağa ekle</button></form><div id="member-result" class="notice"></div><div class="table-wrap"><table><thead><tr><th>Ad</th><th>Subject</th><th>Pozisyon</th><th>Rol</th><th>Durum</th><th></th></tr></thead><tbody id="member-rows"></tbody></table></div><p><button id="topology-save" class="primary" type="button">Bütün mimari taslağını atomik kaydet</button></p><div id="topology-result" class="notice" role="status"></div></article></section>
<section data-tab="events" hidden><article class="panel"><h2>Kurumsal yönetişim olayları</h2><div class="form-grid"><div><label for="event-action">Eylem</label><select id="event-action"><option value="">Tümü</option><option>monitor_view</option><option>topology_read</option><option>topology_change</option><option>membership_change</option><option>retention_inventory_view</option><option>retention_policy_change</option><option>legal_hold_change</option><option>purge_schedule</option><option>purge_execute</option><option>events_view</option></select></div><div><label for="event-decision">Karar</label><select id="event-decision"><option value="">Tümü</option><option>allowed</option><option>denied</option></select></div><div><label for="event-reason">Neden</label><select id="event-reason"><option value="">Tümü</option><option>system_operation</option><option>management_duty</option><option>security_review</option><option>policy_preview</option></select></div><button id="events-load" class="primary" type="button">Filtrele</button></div><div id="event-result" class="notice"></div><div class="table-wrap"><table><thead><tr><th>Zaman</th><th>Eylem</th><th>Karar</th><th>Neden</th><th>Talep</th></tr></thead><tbody id="event-rows"></tbody></table></div><button id="events-more" type="button" disabled>Daha fazla</button></article></section>
<section data-tab="retention" hidden><div class="grid"><article class="panel"><h2>Saklama politikası</h2><label for="retention-days">Arşiv sonrası gün (1–3650)</label><input id="retention-days" type="number" min="1" max="3650"><p class="muted">Revizyon <span id="retention-revision"></span> · epoch <span id="retention-epoch"></span></p><button id="retention-save" class="primary" type="button">CAS ile güncelle</button><div id="retention-result" class="notice"></div></article><article class="panel"><h2>Güvenlik sınırı</h2><p>Bu panel filename, içerik, hash, status note veya candidate id göstermez. Purge yalnız arşivlenmiş, süresi dolmuş, ingest ve legal hold kapıları açık belgelerde çalışır.</p></article></div>
<article class="panel"><h2>Opaque belge yaşam döngüleri</h2><div class="toolbar"><button id="documents-reload" type="button">Yenile</button><button id="documents-more" type="button" disabled>Daha fazla</button><span id="document-result" class="notice"></span></div><div class="table-wrap"><table><thead><tr><th>Belge id</th><th>Durum</th><th>Revizyon</th><th>Aktif hold</th><th>Son purge</th><th></th></tr></thead><tbody id="document-rows"></tbody></table></div></article>
<article id="document-detail" class="panel" hidden><h2>Belge yönetişimi</h2><p class="mono" id="selected-document"></p><p id="selected-status"></p><div class="grid"><div><h3>Legal hold</h3><label for="hold-reason">Kapalı neden</label><select id="hold-reason"><option>litigation</option><option>regulatory</option><option>security_investigation</option></select><button id="hold-create" type="button">Hold oluştur</button><div id="hold-result" class="notice"></div><div class="table-wrap"><table><thead><tr><th>Neden</th><th>Durum</th><th>Rev.</th><th>Zaman</th><th></th></tr></thead><tbody id="hold-rows"></tbody></table></div></div><div><h3>Geri döndürülemez purge</h3><button id="purge-schedule" class="danger" type="button">Purge planla</button><div id="purge-result" class="notice"></div><div class="table-wrap"><table><thead><tr><th>Durum</th><th>Uygunluk</th><th>Deneme</th><th>Hata</th></tr></thead><tbody id="purge-rows"></tbody></table></div></div></div></article></section>
</main><script>__SCRIPT__</script></body></html>""".replace(
    "__STYLE__", PORTAL_STYLE).replace("__SCRIPT__", PORTAL_SCRIPT)


PORTAL_CSP = (
    "default-src 'none'; base-uri 'none'; form-action 'self'; "
    "frame-ancestors 'self'; connect-src 'self'; "
    f"style-src 'sha256-{_source_hash(PORTAL_STYLE)}'; "
    f"script-src 'sha256-{_source_hash(PORTAL_SCRIPT)}'")


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

        def refused(detail="gecersiz portal istegi"):
            return JSONResponse({"detail": detail}, status_code=422)

        def mutation_refused(request):
            if not _mutation_allowed(request):
                return JSONResponse(
                    {"detail": "same-origin portal kaniti gerekli"},
                    status_code=403)
            return None

        async def page(_user=Depends(get_verified_user)):
            return HTMLResponse(PORTAL_HTML, headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": PORTAL_CSP,
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
            })

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
                request: Request, payload: dict = Body(...),
                user=Depends(get_verified_user)):
            rejection = mutation_refused(request)
            if rejection is not None:
                return rejection
            if not _closed_payload(
                    payload,
                    required={"expected_version", "name", "positions",
                              "members"}):
                return refused("gecersiz mimari taslagi")
            status, body = await _proxy(
                user, "PUT", "/v1/org/admin/topology", payload)
            return JSONResponse(body, status_code=status)

        async def member_put(
                identity_id: uuid.UUID, request: Request,
                payload: dict = Body(...), user=Depends(get_verified_user)):
            rejection = mutation_refused(request)
            if rejection is not None:
                return rejection
            required = {"expected_architecture_version",
                        "expected_policy_epoch"}
            optional = {"state", "app_role", "position_id"}
            if (not _closed_payload(payload, required=required,
                                    optional=optional)
                    or not (set(payload) & optional)):
                return refused("gecersiz uyelik degisikligi")
            status, body = await _proxy(
                user, "PUT", "/v1/org/admin/members/" + str(identity_id),
                payload)
            return JSONResponse(body, status_code=status)

        async def reviews(
                reason_code: str = Query(...),
                user=Depends(get_verified_user)):
            if reason_code not in {"management_duty", "security_review"}:
                return JSONResponse({"detail": "gecersiz neden"}, status_code=422)
            status, body = await _proxy(
                user, "GET", "/v1/reviews/queue?limit=100&reason_code=" +
                reason_code)
            return JSONResponse(body, status_code=status)

        async def events(
                limit: int = Query(50, ge=1, le=100),
                action: str | None = Query(default=None),
                decision: str | None = Query(default=None),
                reason_code: str | None = Query(default=None),
                before_created_at: str | None = Query(default=None),
                before_id: uuid.UUID | None = Query(default=None),
                user=Depends(get_verified_user)):
            if (action is not None and action not in EVENT_ACTIONS):
                return refused("gecersiz olay eylemi")
            if decision is not None and decision not in EVENT_DECISIONS:
                return refused("gecersiz olay karari")
            if reason_code is not None and reason_code not in EVENT_REASONS:
                return refused("gecersiz olay nedeni")
            if (before_created_at is None) != (before_id is None):
                return refused("eksik olay imleci")
            query = [("limit", str(limit))]
            for key, value in (("action", action), ("decision", decision),
                               ("reason_code", reason_code),
                               ("before_created_at", before_created_at),
                               ("before_id", before_id)):
                if value is not None:
                    query.append((key, str(value)))
            status, body = await _proxy(
                user, "GET", "/v1/org/admin/audit-events?" +
                urllib.parse.urlencode(query))
            return JSONResponse(body, status_code=status)

        async def review_decision(
                case_id: str, request: Request, payload: dict = Body(...),
                user=Depends(get_verified_user)):
            rejection = mutation_refused(request)
            if rejection is not None:
                return rejection
            allowed = {
                "expected_revision", "expected_policy_epoch", "decision",
                "resolution_code", "reason_code",
            }
            if (set(payload) != allowed
                    or type(payload["expected_revision"]) is not int
                    or payload["expected_revision"] < 1
                    or type(payload["expected_policy_epoch"]) is not int
                    or payload["expected_policy_epoch"] < 1
                    or payload["decision"] not in {"resolved", "dismissed"}
                    or payload["resolution_code"] not in {
                        "corrected", "no_issue", "escalated"}
                    or payload["reason_code"] not in {
                        "management_duty", "security_review"}):
                return JSONResponse(
                    {"detail": "gecersiz inceleme karari"}, status_code=422)
            status, body = await _proxy(
                user, "POST", "/v1/reviews/" + case_id + "/decision", payload)
            return JSONResponse(body, status_code=status)

        async def retention_policy(user=Depends(get_verified_user)):
            status, body = await _proxy(
                user, "GET", "/v1/org/admin/retention-policy")
            return JSONResponse(body, status_code=status)

        async def retention_policy_put(
                request: Request, payload: dict = Body(...),
                user=Depends(get_verified_user)):
            rejection = mutation_refused(request)
            if rejection is not None:
                return rejection
            if not _closed_payload(
                    payload, required={"expected_revision",
                                       "expected_policy_epoch",
                                       "archive_retention_days"}):
                return refused("gecersiz retention politikasi")
            status, body = await _proxy(
                user, "PUT", "/v1/org/admin/retention-policy", payload)
            return JSONResponse(body, status_code=status)

        async def retention_documents(
                limit: int = Query(50, ge=1, le=100),
                before_uploaded_at: str | None = Query(default=None),
                before_id: uuid.UUID | None = Query(default=None),
                user=Depends(get_verified_user)):
            if (before_uploaded_at is None) != (before_id is None):
                return refused("eksik retention belge imleci")
            query = [("limit", str(limit))]
            if before_id is not None:
                query.extend((
                    ("before_uploaded_at", before_uploaded_at),
                    ("before_id", str(before_id)),
                ))
            status, body = await _proxy(
                user, "GET", "/v1/org/admin/retention-documents?" +
                urllib.parse.urlencode(query))
            return JSONResponse(body, status_code=status)

        async def holds(
                document_id: uuid.UUID,
                user=Depends(get_verified_user)):
            status, body = await _proxy(
                user, "GET", f"/documents/{document_id}/legal-holds")
            return JSONResponse(body, status_code=status)

        async def hold_create(
                document_id: uuid.UUID, request: Request,
                payload: dict = Body(...), user=Depends(get_verified_user)):
            rejection = mutation_refused(request)
            if rejection is not None:
                return rejection
            if not _closed_payload(
                    payload, required={"expected_document_revision",
                                       "expected_policy_epoch",
                                       "reason_code"}):
                return refused("gecersiz legal hold")
            status, body = await _proxy(
                user, "POST", f"/documents/{document_id}/legal-holds",
                payload)
            return JSONResponse(body, status_code=status)

        async def hold_release(
                document_id: uuid.UUID, hold_id: uuid.UUID, request: Request,
                payload: dict = Body(...), user=Depends(get_verified_user)):
            rejection = mutation_refused(request)
            if rejection is not None:
                return rejection
            if not _closed_payload(
                    payload, required={"expected_revision",
                                       "expected_policy_epoch"}):
                return refused("gecersiz legal hold kaldirma")
            status, body = await _proxy(
                user, "POST",
                f"/documents/{document_id}/legal-holds/{hold_id}/release",
                payload)
            return JSONResponse(body, status_code=status)

        async def purge_jobs(
                document_id: uuid.UUID,
                user=Depends(get_verified_user)):
            status, body = await _proxy(
                user, "GET", f"/documents/{document_id}/purge-jobs")
            return JSONResponse(body, status_code=status)

        async def purge_schedule(
                document_id: uuid.UUID, request: Request,
                payload: dict = Body(...), user=Depends(get_verified_user)):
            rejection = mutation_refused(request)
            if rejection is not None:
                return rejection
            if not _closed_payload(
                    payload, required={"expected_document_revision",
                                       "expected_policy_epoch"}):
                return refused("gecersiz purge plani")
            status, body = await _proxy(
                user, "POST", f"/documents/{document_id}/purge-jobs",
                payload)
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
        app.add_api_route(PORTAL_PATH + "/api/members/{identity_id}",
                          member_put, methods=["PUT"],
                          include_in_schema=False)
        app.add_api_route(PORTAL_PATH + "/api/reviews", reviews,
                          methods=["GET"], include_in_schema=False)
        app.add_api_route(PORTAL_PATH + "/api/events", events,
                          methods=["GET"], include_in_schema=False)
        app.add_api_route(PORTAL_PATH + "/api/reviews/{case_id}/decision",
                          review_decision, methods=["POST"],
                          include_in_schema=False)
        app.add_api_route(PORTAL_PATH + "/api/retention-policy",
                          retention_policy, methods=["GET"],
                          include_in_schema=False)
        app.add_api_route(PORTAL_PATH + "/api/retention-policy",
                          retention_policy_put, methods=["PUT"],
                          include_in_schema=False)
        app.add_api_route(PORTAL_PATH + "/api/retention-documents",
                          retention_documents, methods=["GET"],
                          include_in_schema=False)
        app.add_api_route(PORTAL_PATH +
                          "/api/documents/{document_id}/holds", holds,
                          methods=["GET"], include_in_schema=False)
        app.add_api_route(PORTAL_PATH +
                          "/api/documents/{document_id}/holds", hold_create,
                          methods=["POST"], include_in_schema=False)
        app.add_api_route(
            PORTAL_PATH +
            "/api/documents/{document_id}/holds/{hold_id}/release",
            hold_release, methods=["POST"], include_in_schema=False)
        app.add_api_route(PORTAL_PATH +
                          "/api/documents/{document_id}/purges", purge_jobs,
                          methods=["GET"], include_in_schema=False)
        app.add_api_route(PORTAL_PATH +
                          "/api/documents/{document_id}/purges",
                          purge_schedule, methods=["POST"],
                          include_in_schema=False)
