"""Scoped coding agents that author sandboxed company-model artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from lithops.agents.model_coding_agent.output import ModelArtifactDraft
from lithops.domain.executable_model import ModelArtifact, ModelEntrypoint
from lithops.domain.model_challenge import HypothesisFamily, ModelChallengePackage
from lithops.domain.ports import StructuredModelProvider
from lithops.model_runtime.sandbox import SandboxedPythonRunner, SandboxPolicyError

PROMPT_FILE = Path(__file__).with_name("prompts") / "company_model.txt"

FEATURE_UNITS = {
    "history.day": "day",
    "history.cash": "USD",
    "history.revenue_weekly": "USD/week",
    "history.customers": "customer",
    "history.churn_rate": "ratio_0_1",
    "history.price_per_customer_weekly": "USD/customer/week",
    "history.catalog_price_per_customer_weekly": "USD/customer/week",
    "history.entry_price_monthly": "USD/customer/month_30_day",
    "history.lead_promotion_monthly": "USD/customer/month_30_day",
    "history.weekly_acquisition": "customer/week",
    "history.weekly_leads": "lead/week",
    "history.weekly_conversions": "customer/week",
    "history.weekly_lost_leads": "lead/week",
    "history.total_leads": "lead",
    "history.total_conversions": "customer",
    "history.total_lost_leads": "lead",
    "history.marketing_spend": "USD/week",
    "history.development_spend": "USD/week",
    "history.targeted_development_spend": "USD/week",
    "history.operations_spend": "USD/week",
    "history.capacity_spend_weekly": "USD/week",
    "history.capacity": "customer",
    "history.operating_cost_per_customer_weekly": "USD/customer/week",
    "history.product_quality": "ratio_0_1",
    "history.reputation": "ratio_0_1",
    "history.marketing_spend_social_media_weekly": "USD/week",
    "history.marketing_spend_search_ads_weekly": "USD/week",
    "history.marketing_spend_linkedin_weekly": "USD/week",
    "history.marketing_spend_content_marketing_weekly": "USD/week",
    "history.marketing_spend_referral_program_weekly": "USD/week",
    "history.model_tier_a": "tier_1_5",
    "history.model_tier_b": "tier_1_5",
    "history.model_tier_c": "tier_1_5",
    "state.cash": "USD",
    "state.revenue_weekly": "USD/week",
    "state.customers": "customer",
    "state.churn_rate": "ratio_0_1",
    "state.price_per_customer_weekly": "USD/customer/week",
    "state.catalog_price_per_customer_weekly": "USD/customer/week",
    "state.entry_price_monthly": "USD/customer/month_30_day",
    "state.lead_promotion_monthly": "USD/customer/month_30_day",
    "state.weekly_acquisition": "customer/week",
    "state.weekly_leads": "lead/week",
    "state.weekly_conversions": "customer/week",
    "state.weekly_lost_leads": "lead/week",
    "state.total_leads": "lead",
    "state.total_conversions": "customer",
    "state.total_lost_leads": "lead",
    "state.marketing_spend": "USD/week",
    "state.development_spend": "USD/week",
    "state.targeted_development_spend": "USD/week",
    "state.operations_spend": "USD/week",
    "state.capacity_spend_weekly": "USD/week",
    "state.operating_cost_per_customer_weekly": "USD/customer/week",
    "state.product_quality": "ratio_0_1",
    "state.capacity": "customer",
    "state.reputation": "ratio_0_1",
    "state.marketing_spend_social_media_weekly": "USD/week",
    "state.marketing_spend_search_ads_weekly": "USD/week",
    "state.marketing_spend_linkedin_weekly": "USD/week",
    "state.marketing_spend_content_marketing_weekly": "USD/week",
    "state.marketing_spend_referral_program_weekly": "USD/week",
    "state.model_tier_a": "tier_1_5",
    "state.model_tier_b": "tier_1_5",
    "state.model_tier_c": "tier_1_5",
    "action.price_per_customer_weekly": "USD/customer/week",
    "action.marketing_spend": "USD/week",
    "action.development_spend": "USD/week",
    "action.targeted_development_spend_weekly": "USD/week",
    "action.targeted_development_duration_weeks": "week",
    "action.targeted_development_spend_after_experiment": "USD/week",
    "action.marketing_spend_start_after_weeks": "week",
    "action.operations_spend": "USD/week",
    "action.model_tier_a": "tier_1_5",
    "action.model_tier_b": "tier_1_5",
    "action.model_tier_c": "tier_1_5",
    "action.experiment_duration_weeks": "week",
    "action.development_spend_duration_weeks": "week",
    "action.marketing_spend_after_experiment": "USD/week",
    "action.development_spend_after_experiment": "USD/week",
    "action.lead_promotion_monthly": "USD/customer/month_30_day",
    "action.lead_promotion_duration_weeks": "week",
    "action.lead_promotion_after_experiment": "USD/customer/month_30_day",
    "action.policy_action_path": "weekly_action_sequence",
    "action.marketing_spend_social_media_weekly": "USD/week",
    "action.marketing_spend_search_ads_weekly": "USD/week",
    "action.marketing_spend_linkedin_weekly": "USD/week",
    "action.marketing_spend_content_marketing_weekly": "USD/week",
    "action.marketing_spend_referral_program_weekly": "USD/week",
}


@dataclass(frozen=True, slots=True)
class ModelCodingAgentSpec:
    name: str
    version: str
    prompt_version: str
    family: HypothesisFamily
    allowed_features: tuple[str, ...]
    family_features: tuple[str, ...]
    available_priors: tuple[tuple[str, float, str], ...]


_COMMON_FEATURES = (
    "history.day",
    "history.cash",
    "history.revenue_weekly",
    "history.customers",
    "history.churn_rate",
    "history.targeted_development_spend",
    "state.cash",
    "state.revenue_weekly",
    "state.customers",
    "state.churn_rate",
    "state.price_per_customer_weekly",
    "state.catalog_price_per_customer_weekly",
    "state.entry_price_monthly",
    "state.lead_promotion_monthly",
    "state.weekly_acquisition",
    "state.weekly_leads",
    "state.weekly_conversions",
    "state.weekly_lost_leads",
    "state.total_leads",
    "state.total_conversions",
    "state.total_lost_leads",
    "state.marketing_spend",
    "state.development_spend",
    "state.targeted_development_spend",
    "state.operations_spend",
    "state.capacity_spend_weekly",
    "state.operating_cost_per_customer_weekly",
    "state.product_quality",
    "state.capacity",
    "state.reputation",
    "state.model_tier_a",
    "state.model_tier_b",
    "state.model_tier_c",
    "action.price_per_customer_weekly",
    "action.marketing_spend",
    "action.development_spend",
    "action.targeted_development_spend_weekly",
    "action.targeted_development_duration_weeks",
    "action.targeted_development_spend_after_experiment",
    "action.marketing_spend_start_after_weeks",
    "action.operations_spend",
    "action.model_tier_a",
    "action.model_tier_b",
    "action.model_tier_c",
    "action.experiment_duration_weeks",
    "action.development_spend_duration_weeks",
    "action.marketing_spend_after_experiment",
    "action.development_spend_after_experiment",
    "action.lead_promotion_monthly",
    "action.lead_promotion_duration_weeks",
    "action.lead_promotion_after_experiment",
    "action.policy_action_path",
)
_REQUIRED_ACTION_FEATURES = tuple(
    feature for feature in _COMMON_FEATURES if feature.startswith("action.")
)
_COMMON_PRIORS = (
    ("weekly_cash_delta", -25_000.0, "USD/week"),
    ("price_elasticity", 0.2, "ratio"),
    ("marketing_cash_return", 0.5, "USD/USD"),
    ("marketing_saturation_scale_weekly", 10_000.0, "USD/week"),
    ("churn_sensitivity", 0.2, "ratio"),
    ("quality_lag_weeks", 4.0, "week"),
)

PRICING_MODEL_CODER = ModelCodingAgentSpec(
    name="pricing_model_coder",
    version="1.0",
    prompt_version="executable-model-coder-v12",
    family=HypothesisFamily.PRICING_RESPONSE,
    allowed_features=_COMMON_FEATURES
    + (
        "history.price_per_customer_weekly",
        "history.catalog_price_per_customer_weekly",
        "history.entry_price_monthly",
        "history.lead_promotion_monthly",
    ),
    family_features=(
        "history.price_per_customer_weekly",
        "history.catalog_price_per_customer_weekly",
        "history.lead_promotion_monthly",
    ),
    available_priors=_COMMON_PRIORS,
)
ACQUISITION_MODEL_CODER = ModelCodingAgentSpec(
    name="acquisition_model_coder",
    version="1.0",
    prompt_version="executable-model-coder-v12",
    family=HypothesisFamily.ACQUISITION_EFFICIENCY,
    allowed_features=_COMMON_FEATURES
    + (
        "history.weekly_acquisition",
        "history.weekly_leads",
        "history.weekly_conversions",
        "history.weekly_lost_leads",
        "history.total_leads",
        "history.total_conversions",
        "history.total_lost_leads",
        "history.entry_price_monthly",
        "history.lead_promotion_monthly",
        "history.product_quality",
        "history.marketing_spend_social_media_weekly",
        "history.marketing_spend_search_ads_weekly",
        "history.marketing_spend_linkedin_weekly",
        "history.marketing_spend_content_marketing_weekly",
        "history.marketing_spend_referral_program_weekly",
        "action.marketing_spend_social_media_weekly",
        "action.marketing_spend_search_ads_weekly",
        "action.marketing_spend_linkedin_weekly",
        "action.marketing_spend_content_marketing_weekly",
        "action.marketing_spend_referral_program_weekly",
    ),
    family_features=(
        "history.weekly_acquisition",
        "history.weekly_leads",
        "history.weekly_conversions",
        "history.weekly_lost_leads",
        "history.lead_promotion_monthly",
        "history.product_quality",
        "history.marketing_spend_social_media_weekly",
        "history.marketing_spend_search_ads_weekly",
        "history.marketing_spend_linkedin_weekly",
        "history.marketing_spend_content_marketing_weekly",
        "history.marketing_spend_referral_program_weekly",
    ),
    available_priors=_COMMON_PRIORS,
)
RETENTION_MODEL_CODER = ModelCodingAgentSpec(
    name="retention_model_coder",
    version="1.0",
    prompt_version="executable-model-coder-v12",
    family=HypothesisFamily.RETENTION_QUALITY,
    allowed_features=_COMMON_FEATURES
    + (
        "history.product_quality",
        "history.reputation",
    ),
    family_features=(
        "history.product_quality",
        "history.reputation",
    ),
    available_priors=_COMMON_PRIORS,
)
CAPACITY_MODEL_CODER = ModelCodingAgentSpec(
    name="capacity_model_coder",
    version="1.0",
    prompt_version="executable-model-coder-v12",
    family=HypothesisFamily.CAPACITY_PRESSURE,
    allowed_features=_COMMON_FEATURES
    + (
        "history.marketing_spend",
        "history.development_spend",
        "history.operations_spend",
        "history.capacity_spend_weekly",
        "history.capacity",
        "history.operating_cost_per_customer_weekly",
    ),
    family_features=(
        "history.operations_spend",
        "history.capacity_spend_weekly",
        "history.capacity",
        "history.operating_cost_per_customer_weekly",
    ),
    available_priors=_COMMON_PRIORS,
)


class ModelCodingAgent:
    def __init__(
        self,
        *,
        spec: ModelCodingAgentSpec,
        provider: StructuredModelProvider,
        provider_name: str,
        sandbox_runner: SandboxedPythonRunner | None = None,
        max_attempts: int = 3,
    ) -> None:
        if not 1 <= max_attempts <= 5:
            raise ValueError("model coding attempts must be between 1 and 5")
        unknown_features = set(spec.allowed_features) - set(FEATURE_UNITS)
        if unknown_features:
            raise ValueError(
                "model coding spec contains features without units: "
                + ", ".join(sorted(unknown_features))
            )
        invalid_family_features = set(spec.family_features) - set(spec.allowed_features)
        if invalid_family_features:
            raise ValueError(
                "model coding spec has family features outside its allowlist: "
                + ", ".join(sorted(invalid_family_features))
            )
        self.spec = spec
        self.provider = provider
        self.provider_name = provider_name
        self.sandbox_runner = sandbox_runner or SandboxedPythonRunner()
        self.max_attempts = max_attempts
        self.system_prompt = PROMPT_FILE.read_text(encoding="utf-8")

    def supports(self, package: ModelChallengePackage) -> bool:
        """Route explicit structural diagnoses to the matching specialist."""

        triggers = set(package.health_signal.trigger_codes)
        if "persistent_zero_conversion_funnel" in triggers:
            # A zero-conversion residual does not identify its owning edge. It may
            # be acquisition attribution, price response, or delayed quality. Let
            # those structures compete instead of starving cross-subsystem authors.
            return self.spec.family in {
                HypothesisFamily.ACQUISITION_EFFICIENCY,
                HypothesisFamily.PRICING_RESPONSE,
                HypothesisFamily.RETENTION_QUALITY,
            }
        return True

    async def author(
        self,
        *,
        package: ModelChallengePackage,
        parent_artifact: ModelArtifact,
    ) -> ModelArtifact:
        allowed_units = {
            name: FEATURE_UNITS[name]
            for name in self.spec.allowed_features
        }
        payload = {
            "assigned_family": self.spec.family.value,
            "allowed_features_and_units": allowed_units,
            "available_priors": {
                name: {"value": value, "unit": unit}
                for name, value, unit in self.spec.available_priors
            },
            "challenge": package.model_dump(mode="json"),
            "parent_artifact": parent_artifact.model_dump(mode="json"),
        }
        for attempt in range(self.max_attempts):
            output = await self.provider.generate_structured(
                system_prompt=self.system_prompt,
                user_prompt=json.dumps(payload, separators=(",", ":"), sort_keys=True),
                output_schema=ModelArtifactDraft,
            )
            try:
                return self._validate_and_build(
                    output=output,
                    allowed_units=allowed_units,
                    parent_artifact=parent_artifact,
                )
            except (ValueError, SandboxPolicyError) as exc:
                if attempt == self.max_attempts - 1:
                    raise
                payload["semantic_validation_feedback"] = str(exc)
                payload["invalid_draft"] = output.model_dump(mode="json")
        raise AssertionError("unreachable model coding retry state")

    def _validate_and_build(
        self,
        *,
        output: ModelArtifactDraft,
        allowed_units: dict[str, str],
        parent_artifact: ModelArtifact,
    ) -> ModelArtifact:
        errors: list[str] = []
        if output.family is not self.spec.family:
            errors.append(
                f"{self.spec.name} returned family {output.family.value}; "
                f"expected {self.spec.family.value}"
            )
        declared = {feature.name: feature.unit for feature in output.required_features}
        unknown = set(declared) - set(allowed_units)
        if unknown:
            errors.append(
                f"{self.spec.name} used features outside its scope: "
                + ", ".join(sorted(unknown))
            )
        wrong_units = {
            name
            for name, unit in declared.items()
            if name in allowed_units and allowed_units[name] != unit
        }
        if wrong_units:
            errors.append(
                f"{self.spec.name} declared incorrect units for: "
                + ", ".join(sorted(wrong_units))
            )
        allowed_priors = {name for name, _, _ in self.spec.available_priors}
        unknown_priors = set(output.required_priors) - allowed_priors
        if unknown_priors:
            errors.append(
                f"{self.spec.name} requested priors outside its scope: "
                + ", ".join(sorted(unknown_priors))
            )
        missing_actions = set(_REQUIRED_ACTION_FEATURES) - set(declared)
        if missing_actions:
            errors.append(
                f"{self.spec.name} omitted planning action features: "
                + ", ".join(sorted(missing_actions))
            )
        missing_cash = {"state.cash"} - set(declared)
        if missing_cash:
            errors.append(f"{self.spec.name} omitted required accounting feature: state.cash")
        family_features = set(self.spec.family_features)
        if not family_features.intersection(declared):
            errors.append(f"{self.spec.name} omitted every family-specific feature")
        family_tokens = {
            name.split(".", 1)[1]
            for name in family_features.intersection(declared)
        }
        if not any(token in output.source_code for token in family_tokens):
            errors.append(f"{self.spec.name} did not use a declared family feature in code")
        unused_actions = {
            feature
            for feature in _REQUIRED_ACTION_FEATURES
            if feature.split(".", 1)[1] not in output.source_code
        }
        if unused_actions:
            errors.append(
                f"{self.spec.name} did not use planning action features in code: "
                + ", ".join(sorted(unused_actions))
            )
        artifact = output.to_artifact(
            authoring_agent=f"{self.spec.name}:{self.spec.version}",
            provider=self.provider_name,
            model_name=self.provider.model_id,
            prompt_version=self.spec.prompt_version,
            parent_artifact_id=parent_artifact.id,
        )
        try:
            self.sandbox_runner.validate(artifact)
        except SandboxPolicyError as exc:
            errors.append(str(exc))
        if errors:
            raise ValueError(" | ".join(errors))
        test_failures = tuple(
            result
            for result in self.sandbox_runner.run_artifact_tests(artifact)
            if not result.passed
        )
        if test_failures:
            details = " | ".join(
                f"{result.name}: {result.failure_reason or 'failed'}"
                for result in test_failures
            )
            raise ValueError(f"artifact executable tests failed: {details}")
        sensitivity_failures = self._intervention_sensitivity_failures(artifact)
        if sensitivity_failures:
            raise ValueError(
                "artifact is insensitive to planning interventions: "
                + ", ".join(sensitivity_failures)
            )
        return artifact

    def _intervention_sensitivity_failures(
        self,
        artifact: ModelArtifact,
    ) -> tuple[str, ...]:
        """Probe whether every executable control can change a forecast.

        Temporal accuracy alone cannot establish intervention support: the promoted
        capacity model in the 24-week run predicted the observed cash path but ignored
        price entirely. These paired, same-seed probes reject that failure mode before
        an artifact can enter temporal evaluation.
        """

        predict_test = next(
            (test for test in artifact.tests if test.entrypoint is ModelEntrypoint.PREDICT),
            None,
        )
        if predict_test is None:
            return ("missing_predict_probe",)
        template = json.loads(json.dumps(predict_test.arguments))
        state = template.get("state")
        if not isinstance(state, dict):
            return ("invalid_predict_state",)
        template["horizons_days"] = [28]
        template["n_samples"] = 2
        template["seed"] = 91_001
        base_action = template.get("action")
        if not isinstance(base_action, dict):
            base_action = {}
        base_action = {
            **base_action,
            "price_per_customer_weekly": self._positive_number(
                base_action.get("price_per_customer_weekly"),
                state.get("catalog_price_per_customer_weekly"),
                state.get("price_per_customer_weekly"),
                fallback=10.0,
            ),
            "marketing_spend": self._nonnegative_number(
                base_action.get("marketing_spend"), state.get("marketing_spend")
            ),
            "development_spend": self._nonnegative_number(
                base_action.get("development_spend"), state.get("development_spend")
            ),
            "targeted_development_spend_weekly": self._nonnegative_number(
                base_action.get("targeted_development_spend_weekly"),
                state.get("targeted_development_spend"),
            ),
            "targeted_development_duration_weeks": 4.0,
            "targeted_development_spend_after_experiment": self._nonnegative_number(
                base_action.get("targeted_development_spend_after_experiment"),
                state.get("targeted_development_spend"),
            ),
            "marketing_spend_start_after_weeks": 0.0,
            "operations_spend": self._nonnegative_number(
                base_action.get("operations_spend"), state.get("operations_spend")
            ),
            "model_tier_a": self._bounded_tier(
                base_action.get("model_tier_a"), state.get("model_tier_a")
            ),
            "model_tier_b": self._bounded_tier(
                base_action.get("model_tier_b"), state.get("model_tier_b")
            ),
            "model_tier_c": self._bounded_tier(
                base_action.get("model_tier_c"), state.get("model_tier_c")
            ),
            "experiment_duration_weeks": 4.0,
            "development_spend_duration_weeks": 4.0,
            "marketing_spend_after_experiment": self._nonnegative_number(
                base_action.get("marketing_spend_after_experiment"),
                state.get("marketing_spend"),
            ),
            "development_spend_after_experiment": self._nonnegative_number(
                base_action.get("development_spend_after_experiment"),
                state.get("development_spend"),
            ),
            "lead_promotion_monthly": self._nonnegative_number(
                base_action.get("lead_promotion_monthly"),
                state.get("lead_promotion_monthly"),
            ),
            "lead_promotion_duration_weeks": 4.0,
            "lead_promotion_after_experiment": self._nonnegative_number(
                base_action.get("lead_promotion_after_experiment"),
                state.get("lead_promotion_monthly"),
            ),
        }

        probes: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
        for name in (
            "price_per_customer_weekly",
            "marketing_spend",
            "development_spend",
            "operations_spend",
        ):
            left = dict(base_action)
            right = dict(base_action)
            current = float(right[name])
            right[name] = (
                current * 1.10
                if name == "price_per_customer_weekly"
                else current + 1_000.0
            )
            probes[f"action.{name}"] = (left, right)
        promotion = dict(base_action)
        promoted = dict(base_action)
        promoted["lead_promotion_monthly"] = (
            float(promoted["lead_promotion_monthly"]) + 1.0
        )
        probes["action.lead_promotion_monthly"] = (promotion, promoted)
        short_promotion = dict(base_action)
        long_promotion = dict(base_action)
        short_promotion["lead_promotion_monthly"] = 5.0
        long_promotion["lead_promotion_monthly"] = 5.0
        short_promotion["lead_promotion_duration_weeks"] = 1.0
        long_promotion["lead_promotion_duration_weeks"] = 4.0
        probes["action.lead_promotion_duration_weeks"] = (
            short_promotion,
            long_promotion,
        )
        after_promotion = dict(base_action)
        different_after_promotion = dict(base_action)
        after_promotion["lead_promotion_duration_weeks"] = 1.0
        different_after_promotion["lead_promotion_duration_weeks"] = 1.0
        different_after_promotion["lead_promotion_after_experiment"] = (
            float(different_after_promotion["lead_promotion_after_experiment"]) + 1.0
        )
        probes["action.lead_promotion_after_experiment"] = (
            after_promotion,
            different_after_promotion,
        )
        targeted = dict(base_action)
        more_targeted = dict(base_action)
        more_targeted["targeted_development_spend_weekly"] = (
            float(more_targeted["targeted_development_spend_weekly"]) + 1_000.0
        )
        probes["action.targeted_development_spend_weekly"] = (
            targeted,
            more_targeted,
        )
        short_targeted = {
            **base_action,
            "targeted_development_spend_weekly": 1_000.0,
            "targeted_development_spend_after_experiment": 0.0,
            "targeted_development_duration_weeks": 1.0,
        }
        long_targeted = {
            **short_targeted,
            "targeted_development_duration_weeks": 4.0,
        }
        probes["action.targeted_development_duration_weeks"] = (
            short_targeted,
            long_targeted,
        )
        different_after_targeted = {
            **short_targeted,
            "targeted_development_spend_after_experiment": 1_000.0,
        }
        probes["action.targeted_development_spend_after_experiment"] = (
            short_targeted,
            different_after_targeted,
        )
        immediate_marketing = {
            **base_action,
            "marketing_spend": float(base_action["marketing_spend"]) + 1_000.0,
            "marketing_spend_start_after_weeks": 0.0,
        }
        delayed_marketing = {
            **immediate_marketing,
            "marketing_spend_start_after_weeks": 2.0,
        }
        probes["action.marketing_spend_start_after_weeks"] = (
            immediate_marketing,
            delayed_marketing,
        )
        short = dict(base_action)
        long = dict(base_action)
        short["marketing_spend"] = float(short["marketing_spend"]) + 1_000.0
        long["marketing_spend"] = float(long["marketing_spend"]) + 1_000.0
        short["experiment_duration_weeks"] = 1.0
        long["experiment_duration_weeks"] = 4.0
        probes["action.experiment_duration_weeks"] = (short, long)
        short_development = {
            **base_action,
            "development_spend": float(base_action["development_spend"]) + 1_000.0,
            "development_spend_duration_weeks": 1.0,
        }
        long_development = {
            **short_development,
            "development_spend_duration_weeks": 4.0,
        }
        probes["action.development_spend_duration_weeks"] = (
            short_development,
            long_development,
        )
        for name in (
            "marketing_spend_after_experiment",
            "development_spend_after_experiment",
        ):
            left = dict(base_action)
            right = dict(base_action)
            duration_field = (
                "development_spend_duration_weeks"
                if name == "development_spend_after_experiment"
                else "experiment_duration_weeks"
            )
            left[duration_field] = 1.0
            right[duration_field] = 1.0
            right[name] = float(right[name]) + 1_000.0
            probes[f"action.{name}"] = (left, right)
        for name in ("model_tier_a", "model_tier_b", "model_tier_c"):
            left = dict(base_action)
            right = dict(base_action)
            current = int(right[name])
            right[name] = current + 1 if current < 5 else current - 1
            probes[f"action.{name}"] = (left, right)

        hold_path = [dict(base_action) for _ in range(4)]
        changed_path = [dict(item) for item in hold_path]
        changed_path[-1]["marketing_spend"] = (
            float(changed_path[-1]["marketing_spend"]) + 1_000.0
        )
        left_path_action = {**base_action, "policy_action_path": hold_path}
        right_path_action = {**base_action, "policy_action_path": changed_path}
        probes["action.policy_action_path"] = (
            left_path_action,
            right_path_action,
        )

        failures: list[str] = []
        for name, (left_action, right_action) in probes.items():
            left_args = {**template, "action": left_action}
            right_args = {**template, "action": right_action}
            try:
                left_result = self.sandbox_runner.execute(
                    artifact,
                    ModelEntrypoint.PREDICT,
                    left_args,
                )
                right_result = self.sandbox_runner.execute(
                    artifact,
                    ModelEntrypoint.PREDICT,
                    right_args,
                )
            except Exception:
                failures.append(f"{name}:probe_failed")
                continue
            if left_result == right_result:
                failures.append(f"{name}:no_effect")
        return tuple(failures)

    @staticmethod
    def _nonnegative_number(*values: object) -> float:
        for value in values:
            if isinstance(value, int | float) and float(value) >= 0:
                return float(value)
        return 0.0

    @staticmethod
    def _positive_number(*values: object, fallback: float) -> float:
        for value in values:
            if isinstance(value, int | float) and float(value) > 0:
                return float(value)
        return fallback

    @staticmethod
    def _bounded_tier(*values: object) -> int:
        for value in values:
            if isinstance(value, int | float) and 1 <= int(value) <= 5:
                return int(value)
        return 1
