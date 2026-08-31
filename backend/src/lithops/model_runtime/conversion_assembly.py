"""Executable company model composed from a typed conversion program and trusted shell."""

from __future__ import annotations

from pydantic import JsonValue

from lithops.domain.component_program import (
    ConversionEvidence,
    ConversionFeature,
    FittedConversionProgram,
)
from lithops.domain.executable_model import (
    CompanyModelFitRequest,
    FittedModel,
    ModelArtifact,
    ModelRuntimeKind,
)
from lithops.domain.world_model import WorldModelVersion
from lithops.model_runtime.baseline import FixedBaselineModel
from lithops.model_runtime.component_program import (
    CompiledConversionComponent,
    fit_conversion_program,
)
from lithops.simulator.components import (
    BASELINE_TRANSITION_ASSEMBLY,
    TransitionModelAssembly,
)


class ConversionAssemblyModel(FixedBaselineModel):
    """Keeps accounting/costs trusted while replacing only conversion response."""

    def __init__(self, artifact: ModelArtifact) -> None:
        if artifact.runtime_kind is not ModelRuntimeKind.TYPED_COMPONENT_ASSEMBLY:
            raise ValueError("conversion assembly requires a typed component artifact")
        if artifact.component_program is None:
            raise ValueError("conversion assembly artifact has no component program")
        self._artifact = artifact

    def fit(self, request: CompanyModelFitRequest) -> FittedModel:
        raw_model = request.prior.get("legacy_world_model")
        if not isinstance(raw_model, dict):
            raise ValueError("conversion assembly requires prior.legacy_world_model")
        world_model = WorldModelVersion.model_validate(raw_model)
        evidence = tuple(
            item for row in request.history if (item := self._evidence_row(row)) is not None
        )
        if not evidence:
            raise ValueError("conversion assembly requires observed lead exposure")
        assert self.artifact.component_program is not None
        fitted_conversion = fit_conversion_program(
            self.artifact.component_program,
            evidence,
        )
        return FittedModel.create(
            artifact=self.artifact,
            request=request,
            fitted_state={
                "legacy_world_model": world_model.model_dump(mode="json"),
                "cash_flow_residual_sigma_weekly": self._cash_flow_residual_sigma(request.history),
                "fitted_conversion": fitted_conversion.model_dump(mode="json"),
            },
        )

    def _transition_assembly(
        self,
        fitted_model: FittedModel,
    ) -> TransitionModelAssembly:
        raw = fitted_model.fitted_state.get("fitted_conversion")
        if not isinstance(raw, dict):
            raise ValueError("fitted conversion assembly is missing component state")
        fitted = FittedConversionProgram.model_validate(raw)
        return TransitionModelAssembly(
            quality=BASELINE_TRANSITION_ASSEMBLY.quality,
            lead_arrival=BASELINE_TRANSITION_ASSEMBLY.lead_arrival,
            conversion=CompiledConversionComponent(fitted),
        )

    def diagnostics(self, fitted_model: FittedModel) -> dict[str, JsonValue]:
        raw = fitted_model.fitted_state.get("fitted_conversion")
        if not isinstance(raw, dict):
            raise ValueError("fitted conversion assembly is missing component state")
        fitted = FittedConversionProgram.model_validate(raw)
        return {
            "baseline": False,
            "assembly": True,
            "component_scope": "conversion",
            "program_name": fitted.program.name,
            "program_link": fitted.program.link.value,
            "fit_log_loss": fitted.fit_log_loss,
            "training_exposures": len(fitted.observation_ids),
        }

    @staticmethod
    def _number(row: dict[str, JsonValue], name: str, default: float = 0.0) -> float:
        value = row.get(name, default)
        return float(value) if isinstance(value, int | float) else default

    @classmethod
    def _evidence_row(
        cls,
        row: dict[str, JsonValue],
    ) -> ConversionEvidence | None:
        leads = cls._number(row, "weekly_leads")
        if leads <= 0:
            return None
        entry_price = cls._number(
            row,
            "entry_price_monthly",
            cls._number(row, "catalog_price_per_customer_weekly", 1.0) * 30.0 / 7.0,
        )
        promotion = cls._number(row, "lead_promotion_monthly")
        day = int(cls._number(row, "day"))
        return ConversionEvidence(
            observation_id=f"observation:{day}",
            day=day,
            leads=leads,
            conversions=min(leads, cls._number(row, "weekly_conversions")),
            features={
                ConversionFeature.PRODUCT_QUALITY: cls._number(row, "product_quality"),
                ConversionFeature.NET_ENTRY_PRICE_MONTHLY: max(
                    0.01,
                    entry_price - promotion,
                ),
                ConversionFeature.REPUTATION: cls._number(row, "reputation"),
                ConversionFeature.MARKETING_SPEND_WEEKLY: cls._number(
                    row,
                    "marketing_spend",
                ),
                ConversionFeature.SOCIAL_MEDIA_SPEND_WEEKLY: cls._number(
                    row,
                    "marketing_spend_social_media_weekly",
                ),
                ConversionFeature.SEARCH_ADS_SPEND_WEEKLY: cls._number(
                    row,
                    "marketing_spend_search_ads_weekly",
                ),
                ConversionFeature.LINKEDIN_SPEND_WEEKLY: cls._number(
                    row,
                    "marketing_spend_linkedin_weekly",
                ),
                ConversionFeature.CONTENT_MARKETING_SPEND_WEEKLY: cls._number(
                    row,
                    "marketing_spend_content_marketing_weekly",
                ),
                ConversionFeature.REFERRAL_PROGRAM_SPEND_WEEKLY: cls._number(
                    row,
                    "marketing_spend_referral_program_weekly",
                ),
            },
        )
