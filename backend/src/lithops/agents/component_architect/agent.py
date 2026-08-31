from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from lithops.domain.component_program import (
    ConversionComponentProgram,
    ConversionFeature,
    ConversionLink,
)
from lithops.domain.executable_model import (
    ModelArtifact,
    ModelFeatureRequirement,
    ModelRuntimeKind,
)
from lithops.domain.model_challenge import ModelChallengePackage
from lithops.domain.ports import StructuredModelProvider

PROMPT_PATH = Path(__file__).with_name("prompt.txt")

_HISTORY_FEATURES: dict[ConversionFeature, tuple[tuple[str, str], ...]] = {
    ConversionFeature.PRODUCT_QUALITY: (("history.product_quality", "ratio"),),
    ConversionFeature.NET_ENTRY_PRICE_MONTHLY: (
        ("history.entry_price_monthly", "USD/customer/month_30_day"),
        ("history.lead_promotion_monthly", "USD/customer/month_30_day"),
    ),
    ConversionFeature.REPUTATION: (("history.reputation", "ratio"),),
    ConversionFeature.MARKETING_SPEND_WEEKLY: (("history.marketing_spend", "USD/week"),),
    ConversionFeature.SOCIAL_MEDIA_SPEND_WEEKLY: (
        ("history.marketing_spend_social_media_weekly", "USD/week"),
    ),
    ConversionFeature.SEARCH_ADS_SPEND_WEEKLY: (
        ("history.marketing_spend_search_ads_weekly", "USD/week"),
    ),
    ConversionFeature.LINKEDIN_SPEND_WEEKLY: (
        ("history.marketing_spend_linkedin_weekly", "USD/week"),
    ),
    ConversionFeature.CONTENT_MARKETING_SPEND_WEEKLY: (
        ("history.marketing_spend_content_marketing_weekly", "USD/week"),
    ),
    ConversionFeature.REFERRAL_PROGRAM_SPEND_WEEKLY: (
        ("history.marketing_spend_referral_program_weekly", "USD/week"),
    ),
}


@dataclass(frozen=True, slots=True)
class ConversionComponentAuthorSpec:
    name: str
    version: str
    prompt_version: str
    required_link: ConversionLink


SMOOTH_CONVERSION_ARCHITECT = ConversionComponentAuthorSpec(
    name="smooth_conversion_architect",
    version="1.0",
    prompt_version="conversion-component-architect-v1",
    required_link=ConversionLink.LOGISTIC,
)
THRESHOLD_CONVERSION_ARCHITECT = ConversionComponentAuthorSpec(
    name="threshold_conversion_architect",
    version="1.0",
    prompt_version="conversion-component-architect-v1",
    required_link=ConversionLink.THRESHOLD_LOGISTIC,
)


class ConversionComponentAuthor:
    """Gemini proposes a small causal structure; Python fits and executes it."""

    def __init__(
        self,
        *,
        spec: ConversionComponentAuthorSpec,
        provider: StructuredModelProvider,
        provider_name: str,
    ) -> None:
        self.spec = spec
        self.provider = provider
        self.provider_name = provider_name
        self.system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    def supports(self, package: ModelChallengePackage) -> bool:
        return "persistent_zero_conversion_funnel" in set(package.health_signal.trigger_codes)

    async def author(
        self,
        *,
        package: ModelChallengePackage,
        parent_artifact: ModelArtifact,
    ) -> ModelArtifact:
        output = await self.provider.generate_structured(
            system_prompt=self.system_prompt,
            user_prompt=json.dumps(
                {
                    "required_link": self.spec.required_link.value,
                    "challenge": package.model_dump(mode="json"),
                    "active_parent": {
                        "artifact_id": str(parent_artifact.id),
                        "artifact_hash": parent_artifact.content_hash,
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            output_schema=ConversionComponentProgram,
        )
        if output.link is not self.spec.required_link:
            raise ValueError(
                f"{self.spec.name} returned {output.link.value}; "
                f"expected {self.spec.required_link.value}"
            )
        required_features = {
            requirement for feature in output.features for requirement in _HISTORY_FEATURES[feature]
        }
        # Cohort counts are always required to fit a conversion likelihood.
        required_features.update(
            {
                ("history.weekly_leads", "count/week"),
                ("history.weekly_conversions", "count/week"),
            }
        )
        return ModelArtifact.create(
            name=output.name.replace("_", "-"),
            protocol_version="2.0",
            runtime_kind=ModelRuntimeKind.TYPED_COMPONENT_ASSEMBLY,
            scope="conversion",
            hypothesis=output.rationale,
            authoring_agent=f"{self.spec.name}:{self.spec.version}",
            provider=self.provider_name,
            model_name=self.provider.model_id,
            prompt_version=self.spec.prompt_version,
            component_program=output,
            required_features=tuple(
                ModelFeatureRequirement(name=name, unit=unit)
                for name, unit in sorted(required_features)
            ),
            limitations=output.falsifiers,
            parent_artifact_id=parent_artifact.id,
        )
