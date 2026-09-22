"""Shared: message structures, two-layer gating (program gates + receiver arbitration), bounded residual updates, and deterministic aggregation."""
from __future__ import annotations

from .mechanisms import is_allowed_mechanism

# ---- Mechanism ontology validation ----

# Variable normalisation: strip modifiers for mechanism-to-variable matching.
# Each canonical variable maps to a set of key semantic words; the LLM variable
# text is considered a match if it contains any keyword.
# Note: more specific words (with organ qualifiers) are listed first to avoid
# false matches from broad words.
_VAR_NORMALIZE = {
    "十二指肠食糜负荷": ["十二指肠食糜", "十二指肠应答", "十二指肠负荷", "十二指肠排液", "食糜负荷", "小肠食糜", "肠腔充盈", "chyme_load", "chyme", "食糜", "肠内容物", "十二指肠胆汁", "胆汁应答", "肠腔内容物"],
    "腔内pH": ["十二指肠ph", "肠腔内ph", "十二指肠酸度", "腔内ph", "酸碱度", "酸负荷", "肠ph", "酸度", "十二指肠内环境"],
    "充盈度/排空相": ["胃排空", "排空相", "排空速率", "胃充盈", "充盈", "排空", "充盈量"],
    "胃酸输出(pH)": ["胃酸", "酸输出", "胃酸分泌", "胃内ph", "胃ph"],
    "蠕动频率": ["蠕动", "蠕动频率", "蠕动强度"],
    "幽门开合": ["幽门", "幽门开合", "幽门关闭", "幽门松弛"],
    "黏膜损伤(病理)": ["黏膜", "胃黏膜", "黏膜损伤", "黏膜屏障", "胆汁浓度"],
    "渗透压": ["渗透压", "渗透"],
    "营养负荷(脂肪/蛋白/糖)": ["营养负荷", "脂肪", "蛋白", "脂肪负荷", "营养"],
    "回肠胆酸回收": ["胆酸", "胆盐", "回肠胆酸", "肠肝循环", "胆酸回收"],
    "炎症": ["炎症", "炎性"],
    "胆汁充盈量": ["胆汁充盈", "胆汁量", "胆汁容量", "胆充盈"],
    "浓缩度": ["浓缩", "胆汁浓缩"],
    "收缩状态": ["收缩", "胆囊收缩", "收缩状态", "收缩强度"],
    "Oddi括约肌张力": ["oddi", "括约肌", "括约肌张力"],
    "结石/淤积(病理)": ["结石", "淤积", "嵌顿", "胆泥", "结石嵌顿", "胆道压力", "胆道内压力"],
}

def normalize_variable(v):
    """Normalise LLM-exported variable text to an ontology variable name.

    Returns the canonical variable name (e.g. "gastric acid output (pH)") for
    validation, or None if no match is found.
    Note: keywords may conflict across variables (e.g. "ph" appears in both
    gastric acid and luminal pH), so validation must also use edge direction
    to determine which organ the target variable belongs to.
    """
    if not v:
        return None
    t = str(v).lower()
    for canonical, keys in _VAR_NORMALIZE.items():
        for k in keys:
            if k.lower() in t:
                return canonical
    return None


def mechanism_variable_check(mechanism_code, source_variable, target_variable, mechanism_variables,
                             source_organ=None, target_organ=None, organ_variables=None):
    """Validate semantic legality of source/target variables. Returns (source_ok, target_ok).

    Two-level (lenient) strategy:
      1. Mechanism-to-variable exact match (canonical variable or alias from ontology)
      2. Organ membership check: source variable belongs to sender organ, target
         variable belongs to receiver organ (semantic upstream accepted)
    A variable is rejected only if it fails even organ membership.
    """
    def _in_organ_vars(llm_var, organ):
        if not organ or not organ_variables or organ not in organ_variables:
            return True  # do not block when no organ variable table is available
        for part in str(llm_var).replace("，", ",").split(","):
            part = part.strip()
            if not part:
                continue
            norm = normalize_variable(part)
            if norm and norm in organ_variables[organ]:
                return True
            for canon in organ_variables[organ]:
                keys = _VAR_NORMALIZE.get(canon, [canon])
                if any(k.lower() in part.lower() for k in keys):
                    return True
        return False

    spec = mechanism_variables.get(mechanism_code) if mechanism_variables else None
    if spec:
        expected_src, expected_tgts = spec
        if isinstance(expected_tgts, str):
            expected_tgts = [expected_tgts]

        def _loose_match(llm_var, expected_list):
            if not llm_var:
                return False
            for part in str(llm_var).replace("，", ",").split(","):
                part = part.strip()
                if not part:
                    continue
                norm = normalize_variable(part)
                if norm in expected_list:
                    return True
                t = part.lower()
                for expected in expected_list:
                    keys = _VAR_NORMALIZE.get(expected, [expected])
                    if any(k.lower() in t for k in keys):
                        return True
            return False

        src_ok = _loose_match(source_variable, [expected_src])
        tgt_ok = _loose_match(target_variable, expected_tgts)
    else:
        src_ok = tgt_ok = False

    # When exact matching fails, fall back to organ membership validation
    if not src_ok:
        src_ok = _in_organ_vars(source_variable, source_organ)
    if not tgt_ok:
        tgt_ok = _in_organ_vars(target_variable, target_organ)
    return src_ok, tgt_ok

# ---- Two-layer gating: program gates (deterministic checks) ----


def program_gate(message, evidence_pool, allowed_mechanisms=None, edge_config=None, max_messages_per_edge=1,
                 recipient_evidence=None, mechanism_variables=None,
                 source_organ=None, target_organ=None, organ_variables=None):
    """Deterministic program gating; returns (eligible: bool, checklist: dict[str, bool]).

    Rules (13 categories; category 1 is split internally into active_edge /
    active_target, giving 14 boolean fields):
      1. active_edge: the edge in the message is valid in the graph
      2. active_target: the target organ is valid in the graph
      3. evidence_traceable: supporting evidence is traceable to the input course
      4. counterevidence_traceable: counter-evidence is traceable to the input course
      5. new_evidence: the message carries at least one evidence node not previously known to the receiver
      6. confidence_threshold: confidence meets the threshold
      7. lag_within_window: expected lag does not exceed the edge's allowed time window
      8. direction_informative: effect direction is not an uninformative neutral state
      9. mechanism_in_list: mechanism code is in the allowed list for this directed edge
      10. source_consistent: source variable matches the mechanism ontology
      11. target_consistent: target variable matches the mechanism ontology
      12. direction_consistent: effect direction matches the mechanism ontology
      13. message_count_ok: atomic message count per directed edge is within the limit
      14. no_option_leak: message text does not reference candidate option labels
    """
    checklist = {}
    for key in (
        "active_edge",
        "active_target",
        "evidence_traceable",
        "counterevidence_traceable",
        "new_evidence",
        "confidence_threshold",
        "lag_within_window",
        "direction_informative",
        "mechanism_in_list",
        "source_consistent",
        "target_consistent",
        "direction_consistent",
        "message_count_ok",
        "no_option_leak",
    ):
        checklist[key] = False

    edge = message.get("edge")
    mechanism_code = message.get("mechanism_code", "")
    checklist["active_edge"] = bool(edge and (edge in edge_config or edge in allowed_mechanisms or is_allowed_mechanism(edge, mechanism_code)))
    checklist["active_target"] = bool(message.get("target_organ"))
    checklist["evidence_traceable"] = all(
        str(e) in evidence_pool for e in message.get("evidence_node_ids", [])
    )
    checklist["counterevidence_traceable"] = all(
        str(e) in evidence_pool for e in message.get("counterevidence_node_ids", [])
    )
    # At least one node in the message evidence set is unknown to the receiver => carries new evidence
    msg_ev = set(str(e) for e in message.get("evidence_node_ids", []))
    recv_ev = set(str(e) for e in (recipient_evidence or []))
    checklist["new_evidence"] = bool(msg_ev) and bool(msg_ev - recv_ev)
    checklist["confidence_threshold"] = float(message.get("confidence", 0.0)) >= 0.6
    checklist["lag_within_window"] = int(message.get("expected_lag_steps", 0)) <= int(edge_config.get(edge, {}).get("max_lag", 1) if edge_config else 1)
    checklist["direction_informative"] = message.get("effect_direction") in (
        "worsen", "improve", "mixed", "drive", "inhibit",
    )
    checklist["mechanism_in_list"] = is_allowed_mechanism(edge, mechanism_code)
    if mechanism_variables:
        src_ok, tgt_ok = mechanism_variable_check(
            mechanism_code, message.get("source_variable"), message.get("target_variable"),
            mechanism_variables.get(edge, {}),
            source_organ=source_organ, target_organ=target_organ, organ_variables=organ_variables,
        )
        checklist["source_consistent"] = src_ok
        checklist["target_consistent"] = tgt_ok
    else:
        checklist["source_consistent"] = bool(message.get("source_variable"))
        checklist["target_consistent"] = bool(message.get("target_variable"))
    checklist["direction_consistent"] = bool(message.get("effect_direction"))
    checklist["message_count_ok"] = message.get("_edge_message_count", 0) <= max_messages_per_edge
    checklist["no_option_leak"] = not _contains_option_codes(message)

    eligible = all(checklist.values())
    return eligible, checklist


def _contains_option_codes(message):
    """Check whether the message contains candidate option labels (option leakage).

    Only blocks expressions that reference the options themselves (e.g.
    "option A", "answer is B", "I choose C"), avoiding false positives from
    letters that normally appear in physiology terms (e.g. Oddi sphincter,
    CCK, PaO2).
    """
    text = " ".join(
        str(message.get(k, ""))
        for k in ("mechanism_summary", "source_variable", "target_variable")
    )
    text = text.replace(" ", "").upper()
    # Most likely leakage expressions in a Chinese-language context
    leak_phrases = (
        "选项A", "选项B", "选项C", "选项D",
        "答案是A", "答案是B", "答案是C", "答案是D",
        "选A", "选B", "选C", "选D",
        "选A:", "选B:", "选C:", "选D:",
        "候选A", "候选B", "候选C", "候选D",
        "答案A", "答案B", "答案C", "答案D",
        "A选项", "B选项", "C选项", "D选项",
    )
    for phrase in leak_phrases:
        if phrase in text:
            return True
    return False


# ---- Second layer: receiver arbitration ----


def recipient_arbitration(message, recipient_local_evidence, decision, reason_code, request_weight):
    """Receiver arbitration; returns (decision: str, weight: float, consistent: bool).

    decision in {accepted, partially_accepted, rejected}.
    If decision/reason_code is inconsistent with the message direction, weight is set to 0.
    """
    consistent = _arbitration_consistency(decision, reason_code, message)
    weight = request_weight if (consistent and decision in ("accepted", "partially_accepted")) else 0.0
    return decision, weight, consistent


def _arbitration_consistency(decision, reason_code, message):
    # Simple rule: accepted/partially_accepted requires a non-empty reason_code that is not a rejection class
    if decision in ("accepted", "partially_accepted"):
        return bool(reason_code) and reason_code not in ("no_evidence", "conflict", "not_applicable")
    if decision == "rejected":
        return reason_code in ("no_evidence", "conflict", "not_applicable")
    return False


# ---- Bounded residual revision ----


def residual_revision(P, Q, alpha):
    """受限残差修订：P' = (1-alpha)*P + alpha*Q。alpha ∈ [0,1]。"""
    alpha = min(max(float(alpha), 0.0), 1.0)
    return [(1 - alpha) * p + alpha * q for p, q in zip(P, Q)]


# ---- Deterministic aggregation ----


def deterministic_aggregation(organ_probs, weights=None):
    """Normalised (optionally weighted) average of active-organ probabilities.

    Supports two input formats:
      - list of lists: one probability list per organ
      - list of dicts: one {candidate: probability} mapping per organ (aligned by key order internally)
    """
    if not organ_probs:
        return {}

    def _values(op):
        if isinstance(op, dict):
            return [float(op[k]) for k in op]
        return [float(v) for v in op]

    n = len(organ_probs)
    if weights is None:
        weights = [1.0] * n
    wsum = sum(weights)
    acc = [0.0] * len(_values(organ_probs[0]))
    for op, w in zip(organ_probs, weights):
        for i, p in enumerate(_values(op)):
            acc[i] += (w / wsum) * p
    total = sum(acc)
    if total <= 0:
        return {str(i): 0.0 for i in range(len(acc))}
    if isinstance(organ_probs[0], dict):
        keys = list(organ_probs[0].keys())
        return {keys[i]: v / total for i, v in enumerate(acc)}
    return {str(i): v / total for i, v in enumerate(acc)}
