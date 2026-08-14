"""PACKAGE B4-R5 -- the plan-only authentication gate.

NO REAL MODEL IS CALLED HERE, and none is needed: every binary is a stub
written into `tmp_path` that answers a status query and COUNTS how many
times it was started. That counter is what makes the negative tests mean
something -- "refused" and "refused before anything ran" are different
claims, and only the second one keeps an API key from reaching a CLI.

NO REAL CREDENTIAL APPEARS IN THIS FILE. The environment values are
obvious fictions, and the accepted status document carries no email and
no organisation id: a fixture that embedded a real one would put it in
the repository forever.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from tools.agent_loop import contract, plan_auth

_STUB = '''\
import json, os, sys, time
from pathlib import Path

CONFIG = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
ARGV = sys.argv[2:]

counter = Path(CONFIG["counter"])
previous = int(counter.read_text(encoding="ascii")) if counter.exists() else 0
counter.write_text(str(previous + 1), encoding="ascii")

mode = CONFIG["mode"]
if mode == "timeout":
    time.sleep(CONFIG.get("seconds", 30))
    sys.exit(0)
if mode == "oversized":
    sys.stdout.write("x" * CONFIG.get("bytes", 200000))
    sys.exit(0)
if mode == "nonzero":
    sys.stdout.write(CONFIG.get("body", ""))
    sys.exit(CONFIG.get("code", 1))
if CONFIG.get("stream") == "stderr":
    sys.stderr.write(CONFIG["body"])
else:
    sys.stdout.write(CONFIG["body"])
sys.exit(0)
'''

PRO = {"loggedIn": True, "authMethod": "claude.ai",
       "apiProvider": "firstParty", "subscriptionType": "pro"}
MAX_PLAN = dict(PRO, subscriptionType="max")
CHATGPT = "Logged in using ChatGPT\n"


def _stub(tmp_path, name, **config):
    holder = tmp_path / "sahte-bin"
    holder.mkdir(exist_ok=True)
    helper = holder / "yardimci.py"
    helper.write_text(_STUB, encoding="utf-8")
    counter = holder / f"{name}-sayac.txt"
    settings = holder / f"{name}.json"
    settings.write_text(json.dumps(dict(config, counter=str(counter))),
                        encoding="utf-8")
    if os.name == "nt":
        shim = holder / f"{name}.cmd"
        shim.write_text(
            f'@echo off\r\n"{sys.executable}" "{helper}" "{settings}" %*\r\n',
            encoding="ascii")
    else:
        shim = holder / f"{name}.sh"
        shim.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{helper}" "{settings}" "$@"\n',
            encoding="ascii")
        shim.chmod(0o755)
    return shim, counter


def _launches(counter):
    return int(counter.read_text(encoding="ascii")) if counter.exists() else 0


@pytest.fixture(autouse=True)
def no_vendor_keys(monkeypatch):
    """The machine running the suite may legitimately hold a key. Every
    test starts from a known-clean environment so a developer's own
    setup cannot turn a red test green or the reverse."""
    for name in plan_auth.FORBIDDEN_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def pair(tmp_path):
    """A healthy implementer/evaluator pair, plus their counters."""
    implementer, first = _stub(tmp_path, "claude", mode="body",
                               body=json.dumps(PRO))
    evaluator, second = _stub(tmp_path, "codex", mode="body",
                              stream="stderr", body=CHATGPT)
    return implementer, first, evaluator, second


def _assert(implementer, evaluator):
    return plan_auth.assert_plan_only(implementer_binary=implementer,
                                      evaluator_binary=evaluator)


# =====================================================================
# A. THE SHAPES THAT ARE ACCEPTED
# =====================================================================

def test_a_claude_pro_subscription_and_a_chatgpt_login_are_accepted(pair):
    """POSITIVE CONTROL: a gate that refused everything would pass every
    negative test in this file."""
    implementer, first, evaluator, second = pair
    result = _assert(implementer, evaluator)
    assert result.implementer_auth == plan_auth.CLAUDE_SUBSCRIPTION
    assert result.implementer_plan == "pro"
    assert result.evaluator_auth == plan_auth.CHATGPT_SUBSCRIPTION
    assert _launches(first) == 1 and _launches(second) == 1
    # nothing but closed values crosses
    assert "@" not in repr(result) and "claude.ai" not in repr(result)


def test_a_max_plan_is_accepted_and_reported_as_itself(tmp_path):
    implementer, _ = _stub(tmp_path, "claude", mode="body",
                           body=json.dumps(MAX_PLAN))
    evaluator, _ = _stub(tmp_path, "codex", mode="body", stream="stderr",
                         body=CHATGPT)
    assert _assert(implementer, evaluator).implementer_plan == "max"


# =====================================================================
# B. THE ENVIRONMENT IS JUDGED BEFORE ANY PROCESS EXISTS
# =====================================================================

@pytest.mark.parametrize("variable", plan_auth.FORBIDDEN_ENV)
def test_an_api_key_refuses_before_a_single_status_command_runs(
        pair, monkeypatch, variable):
    """THE ORDERING IS THE GUARANTEE. A key in the environment must stop
    the run before a CLI is started at all -- the counters prove that,
    and a refusal that arrived after the first launch would still be a
    refusal but not this one."""
    implementer, first, evaluator, second = pair
    monkeypatch.setenv(variable, "kurgu-anahtar-degeri")
    with pytest.raises(plan_auth.PlanAuthRefused) as refused:
        _assert(implementer, evaluator)
    assert _launches(first) == 0 and _launches(second) == 0
    assert refused.value.reason == contract.StopReason.PREFLIGHT_FAILED
    # the NAME may be reported; the value never
    assert "kurgu-anahtar-degeri" not in str(refused.value)
    assert variable in str(refused.value)


def test_a_whitespace_only_key_is_not_a_key(pair, monkeypatch):
    """An empty or blank variable is a leftover, not a credential:
    refusing it would make the gate impossible to satisfy on a machine
    that once exported one."""
    implementer, first, evaluator, _ = pair
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    _assert(implementer, evaluator)
    assert _launches(first) == 1


# =====================================================================
# C. WHAT THE CLIs MAY SAY
# =====================================================================

@pytest.mark.parametrize(
    "payload",
    [dict(PRO, apiProvider="anthropic"),
     dict(PRO, authMethod="apiKey"),
     dict(PRO, subscriptionType="free"),
     dict(PRO, subscriptionType="enterprise"),
     dict(PRO, loggedIn=False),
     dict(PRO, loggedIn="true"),
     {"loggedIn": True}],
    ids=["gateway-saglayici", "api-anahtari", "ucretsiz", "kurumsal",
         "oturum-kapali", "dogru-degil-metin", "eksik-alanlar"])
def test_any_authentication_that_is_not_a_subscription_is_refused(
        tmp_path, payload):
    """`loggedIn: "true"` is the reason this compares to `True` by
    identity: every one of these is truthy, and only one is a boolean."""
    implementer, first = _stub(tmp_path, "claude", mode="body",
                               body=json.dumps(payload))
    evaluator, second = _stub(tmp_path, "codex", mode="body",
                              stream="stderr", body=CHATGPT)
    with pytest.raises(plan_auth.PlanAuthRefused):
        _assert(implementer, evaluator)
    assert _launches(first) == 1
    # the evaluator is never asked once the implementer has failed
    assert _launches(second) == 0


@pytest.mark.parametrize(
    "config",
    [{"mode": "body", "body": "bu JSON degil"},
     {"mode": "body", "body": "[]"},
     {"mode": "body", "body": ""},
     {"mode": "nonzero", "body": json.dumps(PRO), "code": 1},
     {"mode": "oversized"},
     {"mode": "timeout", "seconds": 30}],
    ids=["bozuk-json", "dizi", "bos", "sifirdan-farkli", "tasan", "suresi-dolan"])
def test_a_status_answer_that_cannot_be_read_is_refused(tmp_path, config):
    """Malformed, oversized, non-zero and hung answers are all the same
    verdict: this machine has not PROVEN it is on a subscription."""
    implementer, first = _stub(tmp_path, "claude", **config)
    evaluator, _ = _stub(tmp_path, "codex", mode="body", stream="stderr",
                         body=CHATGPT)
    with pytest.raises(plan_auth.PlanAuthRefused):
        _assert(implementer, evaluator)
    assert _launches(first) == 1, "senaryo kurulmadi: ikili hic calismadi"


@pytest.mark.parametrize(
    "body",
    ["Logged in using an API key\n",
     "Logged in using ChatGPT and an API key\n",
     "Logged in using ChatGPT\nBilling: pay-as-you-go\n",
     "logged in using chatgpt\n",
     "\n"],
    ids=["api-anahtari", "ikisi-birden", "ikinci-satir", "kucuk-harf", "bos"])
def test_an_evaluator_login_that_is_not_exactly_chatgpt_is_refused(
        tmp_path, body):
    """EXACTLY ONE meaningful line, matched exactly. A second line is a
    second claim, and 'ChatGPT and an API key' contains the accepted
    sentence without meaning it."""
    implementer, _ = _stub(tmp_path, "claude", mode="body",
                           body=json.dumps(PRO))
    evaluator, second = _stub(tmp_path, "codex", mode="body",
                              stream="stderr", body=body)
    with pytest.raises(plan_auth.PlanAuthRefused):
        _assert(implementer, evaluator)
    assert _launches(second) == 1


# =====================================================================
# D. THE CHECK ITSELF LEAVES NOTHING BEHIND
# =====================================================================

def test_an_unprovable_cleanup_is_its_own_stronger_refusal(pair, monkeypatch):
    """'You are on the wrong plan' is a decision; 'a process this loop
    started is still alive' is an incident. They must not share a
    class."""
    implementer, _, evaluator, _ = pair
    monkeypatch.setattr(plan_auth, "join_within", lambda *a, **k: False)
    with pytest.raises(plan_auth.PlanAuthCleanupFailed) as refused:
        _assert(implementer, evaluator)
    assert isinstance(refused.value, plan_auth.PlanAuthRefused)
    assert refused.value.cleanup_complete is False


def test_the_probe_leaves_no_temporary_directory_behind(pair):
    """Every probe builds its own working directory so a status query
    cannot touch the checkout -- and removes it, whatever happened."""
    implementer, _, evaluator, _ = pair
    holder = Path(os.environ.get("TEMP", "/tmp"))
    before = {p.name for p in holder.glob("agent-loop-auth-*")}
    _assert(implementer, evaluator)
    after = {p.name for p in holder.glob("agent-loop-auth-*")}
    assert after == before


def test_no_status_output_or_path_reaches_the_refusal(tmp_path):
    """The refusal text is chosen here, not captured: an account name in
    a status document must not travel into a report."""
    sentinel = "GIZLI-HESAP-ADI"
    implementer, _ = _stub(tmp_path, "claude", mode="body", body=json.dumps(
        dict(PRO, subscriptionType="free", email=sentinel)))
    evaluator, _ = _stub(tmp_path, "codex", mode="body", stream="stderr",
                         body=CHATGPT)
    with pytest.raises(plan_auth.PlanAuthRefused) as refused:
        _assert(implementer, evaluator)
    blob = str(refused.value) + repr(refused.value.__dict__)
    assert sentinel not in blob
    assert str(implementer) not in blob and "sahte-bin" not in blob
