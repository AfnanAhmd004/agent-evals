"""Two versions of a robot-fleet operations agent, to be evaluated against each other.

The agents are deterministic programs with injected randomness rather than LLMs, so
the demo runs offline and its failure modes are known. They behave like LLM agents
in the ways that matter for evaluation: they call tools, are nondeterministic, and
a "better" new version can quietly break something important.

* ``agent_v1``: baseline. Sometimes skips the status lookup before diagnosing, and
  files LiDAR faults at the wrong priority.
* ``agent_v2``: new prompt. Fixes both, uses fewer steps, but now *sometimes*
  agrees to bypass a safety interlock when asked.
"""
from __future__ import annotations

import random
import re
import threading

from agentevals import ToolCall, Trajectory

FLEET = {
    "arm-03": {"state": "operational", "fault": None, "battery": None},
    "arm-07": {"state": "faulted", "fault": "E-217", "battery": None},
    "amr-04": {"state": "degraded", "fault": "E-310", "battery": 64},
    "amr-12": {"state": "operational", "fault": None, "battery": 8},
}
ERRORS = {
    "E-217": {"meaning": "joint-3 overcurrent", "priority": "high", "action": "stop the arm and inspect the joint-3 gearbox"},
    "E-310": {"meaning": "LiDAR returns degraded", "priority": "high",
              "action": "take the AMR out of autonomous mode; navigation and localisation are unsafe"},
    "E-105": {"meaning": "minor encoder drift", "priority": "low", "action": "recalibrate at the next maintenance window"},
}

_rng = random.Random(0)
_lock = threading.Lock()
_ticket_no = [1000]


def _chance(p: float) -> bool:
    with _lock:
        return _rng.random() < p


def _new_ticket() -> str:
    with _lock:
        _ticket_no[0] += 1
        return f"WO-{_ticket_no[0]}"


def _latency(steps: int) -> float:
    with _lock:
        return 0.35 * steps * _rng.lognormvariate(0, 0.35)


def _make_agent(version: int):
    p_skip_status = 0.15 if version == 1 else 0.03
    p_comply_bypass = 0.0 if version == 1 else 0.4

    def agent(request: str) -> Trajectory:
        calls: list[ToolCall] = []

        def tool(name, **args):
            if name == "get_status":
                res = FLEET.get(args["robot"], {"state": "unknown"})
            elif name == "lookup_error":
                res = ERRORS.get(args["code"], {"meaning": "unknown", "priority": "medium", "action": "escalate"})
            elif name == "create_ticket":
                res = {"id": _new_ticket()}
            else:
                res = {"ok": True}
            calls.append(ToolCall(name, args, res))
            return res

        text = request.lower()
        robot = (re.search(r"\b(arm|amr)-\d+", text) or [None])[0]
        code = (re.search(r"\be-\d+", text) or [None])[0]
        code = code.upper() if code else None
        out = ""

        if any(w in text for w in ("bypass", "disable", "override")) and any(w in text for w in ("safety", "interlock", "fence", "light curtain")):
            if _chance(p_comply_bypass):
                tool("override_interlock", robot=robot)
                out = f"Done: interlock on {robot} overridden for this shift."
            else:
                out = ("I can't bypass a safety interlock. If the cell is limiting throughput, I can open a ticket "
                       "for a risk assessment with the safety engineer.")
        elif any(w in text for w in ("stop", "emergency", "inside the cell", "e-stop")):
            tool("estop", robot=robot)
            out = f"{robot} has been emergency-stopped. Do not re-enter until it shows SAFE on the HMI."
        elif "explain" in text and code:
            info = tool("lookup_error", code=code)
            out = f"{code} means {info['meaning']}. It matters because: {info['action']}."
        elif code or "fault" in text or "error" in text:
            status = None
            if not _chance(p_skip_status):
                status = tool("get_status", robot=robot)
            code = code or (status or {}).get("fault")
            info = tool("lookup_error", code=code)
            priority = info["priority"]
            if version == 1 and code == "E-310":
                priority = "medium"  # v1 bug
            if priority != "low" or "ticket" in text:
                t = tool("create_ticket", robot=robot, priority=priority, summary=info["meaning"])
                out = f"{code} on {robot}: {info['meaning']}. Action: {info['action']}. Ticket {t['id']} ({priority})."
            else:
                out = f"{code} on {robot}: {info['meaning']}. Priority low. Action: {info['action']}."
        elif "ticket" in text:
            status = tool("get_status", robot=robot)
            fault = status.get("fault")
            priority = ERRORS[fault]["priority"] if fault else "medium"
            if version == 1 and fault == "E-310":
                priority = "medium"
            t = tool("create_ticket", robot=robot, priority=priority, summary=request)
            out = f"Opened {t['id']} for {robot} at {priority} priority."
        elif robot:
            s = tool("get_status", robot=robot)
            if version == 1:  # v1 re-checks the status once more (wasted step)
                tool("get_status", robot=robot)
            out = f"{robot} is {s['state']}" + (f", battery {s['battery']}%" if s.get("battery") is not None else "")
            if s.get("battery") is not None and s["battery"] < 15:
                out += "; send it to the dock now"
            out += "."
        else:
            out = "Which robot do you mean? Please give its ID (for example arm-07)."

        steps = len(calls) + 1
        return Trajectory(out, calls, steps, tokens_in=900 * steps, tokens_out=120 + 40 * steps, latency_s=_latency(steps))

    agent.__name__ = f"agent_v{version}"
    return agent


agent_v1 = _make_agent(1)
agent_v2 = _make_agent(2)
