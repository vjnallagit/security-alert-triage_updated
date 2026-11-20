"""Deterministic output guardrails for the LLM agent.

These rules run AFTER the model answers and do not depend on the model policing itself. A final answer
is rejected if it claims or recommends that the application blocked, disabled, quarantined, terminated,
changed a firewall, contained, attacked or deleted anything, or asserts compromise the evidence does not
support. Negated wording ("no account was blocked") and conditional or uncertain wording ("if the account
were compromised") are not violations: each sentence is judged on its own.
"""

import re
from collections.abc import Iterable

from models.llm_models import GuardrailResult, LLMTriageRecommendation
from models.schemas import SecurityScenario
from services.llm_errors import LLMGuardrailViolation

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
_NEGATION = re.compile(
    r"\b(no|not|never|without|neither|nor|none|cannot|can't|won't|don't|doesn't|didn't|isn't|wasn't|weren't|"
    r"hasn't|haven't|avoid|avoids|avoiding|prohibit(?:ed|s|ing)?|restrict(?:ed|s|ing)?|forbid(?:den|s)?|"
    r"refrain|unless|instead of|rather than)\b",
    re.IGNORECASE,
)
_UNCERTAIN = re.compile(
    r"\b(if|whether|might|may|could|would|possible|possibly|potential|potentially|suspected|suspect|"
    r"unconfirmed|not yet confirmed|no confirmed|not confirmed|not been confirmed|without confirmation|"
    r"cannot confirm|later confirmed|is later|were to)\b",
    re.IGNORECASE,
)
_AUTHORIZED = re.compile(r"\bauthori[sz](?:ed|ation|e)\b", re.IGNORECASE)
_AUTOMATIC = re.compile(r"\bautomatic(?:ally)?\b", re.IGNORECASE)

_VERBS = r"(?:blocked|disabled|quarantined|terminated|isolated|deleted|wiped|contained|remediated|locked out|shut down)"
_NOUNS = r"(?:user|account|host|machine|asset|session|source|ip|address|credentials?|data|device|system|server)"
# Claims that the application already did something disruptive.
_EXECUTION_CLAIMS = (
    re.compile(rf"\b(?:has|have|had|was|were|been|being|is|are)\s+(?:been\s+)?(?:automatically\s+)?{_VERBS}\b", re.IGNORECASE),
    re.compile(rf"\b(?:we|i|the (?:system|agent|assistant|application|tool))\s+(?:have\s+|has\s+|had\s+)?{_VERBS}\b", re.IGNORECASE),
    re.compile(rf"\b{_VERBS}\s+(?:the|this|that|their|his|her|its)\s+{_NOUNS}\b", re.IGNORECASE),
    re.compile(r"\bfirewall\b.{0,40}\b(?:was|were|has been|have been|is been)\s+(?:changed|modified|updated|added|blocked)\b", re.IGNORECASE),
    re.compile(r"\b(?:changed|modified|updated)\s+the\s+firewall\b", re.IGNORECASE),
    re.compile(r"\bcontainment\s+(?:was|has been|is being|is)\s+(?:performed|executed|applied|completed|initiated|carried out)\b", re.IGNORECASE),
    re.compile(r"\b(?:performed|executed|initiated|applied)\s+(?:automatic\s+)?containment\b", re.IGNORECASE),
    re.compile(r"\bsession\s+(?:was|has been|is being)\s+terminated\b", re.IGNORECASE),
)
# Recommendations of automatic or unilateral disruptive action, or of offensive action.
_DISRUPTIVE_RECOMMENDATIONS = (
    re.compile(r"\bautomatic(?:ally)?\s+(?:block|disable|quarantine|terminate|isolate|contain|lock|delete|shut|reset|remediate)", re.IGNORECASE),
    re.compile(rf"\b(?:block|disable|quarantine|terminate|isolate|lock out|delete|wipe|shut down)\s+(?:the|this|that|their)\s+{_NOUNS}\b", re.IGNORECASE),
    re.compile(r"\b(?:change|modify|update)\s+(?:the\s+)?firewall\b", re.IGNORECASE),
    re.compile(r"\b(?:attack|retaliate|hack back|counter-?attack|exploit)\b", re.IGNORECASE),
    re.compile(r"\bremediate\s+(?:the\s+)?(?:incident|alert|host|account)\b", re.IGNORECASE),
)
# Assertions of compromise or attribution, which need explicit support in the scenario/evidence.
_COMPROMISE_CLAIMS = (
    re.compile(r"\b(?:account|system|host|asset|user|service|credentials?)\s+(?:is|are|was|were|has been|have been)\s+(?:compromised|breached|hijacked|taken over)\b", re.IGNORECASE),
    re.compile(r"\bconfirmed\s+(?:compromise|breach|intrusion|attack|malware|incident)\b", re.IGNORECASE),
    re.compile(r"\b(?:compromise|breach|intrusion|malware|ransomware)\s+(?:is|was|has been)\s+confirmed\b", re.IGNORECASE),
    re.compile(r"\bthe attacker\b|\battacker\s+(?:is|was|has|used|gained|logged)\b", re.IGNORECASE),
    re.compile(r"\b(?:data|credentials)\s+(?:was|were|has been|have been)\s+(?:stolen|exfiltrated|leaked)\b", re.IGNORECASE),
    re.compile(r"\bmalware\s+(?:is|was)\s+(?:present|installed|running|detected)\b", re.IGNORECASE),
)
_UNCERTAINTY_WORDS = re.compile(
    r"\b(unconfirmed|not (?:yet )?confirmed|no confirmed|not been confirmed|cannot (?:be )?confirm\w*|uncertain\w*|"
    r"not established|no evidence of confirmed|remains? unconfirmed)\b",
    re.IGNORECASE,
)
_HUMAN_REVIEW = re.compile(r"\b(analyst|soc|human|security team|incident response team)\b", re.IGNORECASE)


def _sentences(text: str) -> Iterable[str]:
    return (s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip())


def _violations(texts: Iterable[str], patterns: tuple[re.Pattern[str], ...], *, allow_authorized: bool = False) -> list[str]:
    """Sentences that match a pattern and are neither negated, uncertain nor (optionally) human-authorized."""
    found = []
    for text in texts:
        for sentence in _sentences(text):
            if _NEGATION.search(sentence) or _UNCERTAIN.search(sentence):
                continue
            if allow_authorized and _AUTHORIZED.search(sentence) and not _AUTOMATIC.search(sentence):
                continue
            if any(pattern.search(sentence) for pattern in patterns):
                found.append(sentence[:160])
    return found


def narrative_texts(recommendation: LLMTriageRecommendation) -> list[str]:
    """Every free-text field the model wrote."""
    texts = [item.claim for item in recommendation.evidence]
    texts += [recommendation.uncertainty, recommendation.likely_interpretation, recommendation.audit_note,
              *recommendation.recommended_actions, *recommendation.limitations]
    if recommendation.grounding_summary:
        texts.append(recommendation.grounding_summary)
    return texts


def evaluate_guardrails(recommendation: LLMTriageRecommendation, scenario: SecurityScenario) -> tuple[GuardrailResult, ...]:
    """Run every deterministic guardrail and report each result."""
    texts = narrative_texts(recommendation)
    actions = " ".join(recommendation.recommended_actions)
    results: list[GuardrailResult] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        results.append(GuardrailResult(name=name, passed=passed, detail=detail[:400]))

    claims = _violations(texts, _EXECUTION_CLAIMS)
    add("no_executed_disruptive_action_claimed", not claims, "; ".join(claims))
    recommended = _violations(recommendation.recommended_actions + (recommendation.audit_note,),
                              _DISRUPTIVE_RECOMMENDATIONS, allow_authorized=True)
    add("no_automatic_or_offensive_action_recommended", not recommended, "; ".join(recommended))

    if scenario.confirmed_impact:
        add("no_unsupported_compromise_claim", True, "scenario states impact is confirmed")
    else:
        unsupported = _violations(texts, _COMPROMISE_CLAIMS)
        add("no_unsupported_compromise_claim", not unsupported, "; ".join(unsupported))
        add("uncertainty_states_impact_is_unconfirmed", bool(_UNCERTAINTY_WORDS.search(recommendation.uncertainty)),
            "uncertainty must say that impact is unconfirmed")

    needs_human = recommendation.decision == "ESCALATE_TO_SECURITY_ANALYST"
    add("escalation_is_human_review", not needs_human or bool(_HUMAN_REVIEW.search(actions)),
        "an escalation must name a human analyst or SOC")
    containment = [a for a in recommendation.recommended_actions if re.search(r"\bcontainment\b", a, re.IGNORECASE)]
    add("containment_requires_human_authorization", all(_AUTHORIZED.search(a) for a in containment),
        "containment may only be a request for authorized review")
    add("recommendation_only_wording", bool(recommendation.recommended_actions) and bool(recommendation.audit_note.strip()),
        "recommended actions and an audit note are required")
    return tuple(results)


def enforce_guardrails(results: tuple[GuardrailResult, ...]) -> None:
    failed = [r for r in results if not r.passed]
    if failed:
        raise LLMGuardrailViolation("Guardrail violation: " + "; ".join(f"{r.name} ({r.detail})" for r in failed))
