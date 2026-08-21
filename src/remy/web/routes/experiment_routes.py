"""REST API for the private multi-model Experiment Lab."""

from __future__ import annotations

from pathlib import Path
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from remy.core.experiment_lab import get_experiment_engine

router = APIRouter()


class ExperimentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    problem: str = Field(min_length=1, max_length=20_000)
    success_criteria: str = Field(default="", max_length=10_000)
    models: list[str] = Field(min_length=1, max_length=8)
    rounds: int = Field(default=2, ge=1, le=5)
    max_calls: int = Field(default=0, ge=0, le=100)
    domain: str = Field(default="general", max_length=40)


class ExperimentData(BaseModel):
    name: str = Field(default="Notes", max_length=200)
    content: str = Field(min_length=1, max_length=80_000)


class ExperimentContinue(BaseModel):
    question: str = Field(min_length=1, max_length=20_000)
    rounds: int = Field(default=1, ge=1, le=3)


class ScenarioIntervention(BaseModel):
    content: str = Field(min_length=1, max_length=10_000)
    round: int | None = Field(default=None, ge=1, le=5)


class ExperimentUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    problem: str = Field(min_length=1, max_length=20_000)
    success_criteria: str = Field(default="", max_length=10_000)
    models: list[str] = Field(min_length=1, max_length=8)
    rounds: int = Field(default=2, ge=1, le=5)
    max_calls: int = Field(default=0, ge=0, le=100)
    domain: str = Field(default="general", max_length=40)


class CanvasPayload(BaseModel):
    canvas: dict


class CanvasTemplateRequest(BaseModel):
    template_id: str = "scientific-panel"
    models: list[str] = Field(default_factory=list, max_length=8)


class SelfModificationProposalPayload(BaseModel):
    candidate_text: str = Field(min_length=20, max_length=4_000)
    rationale: str = Field(default="", max_length=4_000)
    source: str = Field(default="operator", pattern="^(operator|agent|experiment)$")
    source_session_id: str = Field(default="", max_length=160)


class SelfModificationEvalPayload(BaseModel):
    matrix_id: str = Field(min_length=1, max_length=160)


class SelfModificationEvalRunPayload(BaseModel):
    case_ids: list[str] = Field(min_length=3, max_length=25)
    preferred_model: str = Field(default="", max_length=240)
    name: str = Field(default="", max_length=160)


class SelfModificationApprovalPayload(BaseModel):
    confirm_candidate_hash: str = Field(min_length=64, max_length=64)
    approved_by: str = Field(min_length=1, max_length=160)


class SelfModificationCanaryPayload(BaseModel):
    confirm_candidate_hash: str = Field(min_length=64, max_length=64)
    canary_percent: int = Field(default=10, ge=5, le=25)


class SelfModificationCanaryEvaluationPayload(BaseModel):
    candidate_requests: int = Field(ge=0, le=1_000_000)
    baseline_requests: int = Field(ge=0, le=1_000_000)
    candidate_failure_rate: float = Field(ge=0, le=1)
    baseline_failure_rate: float = Field(ge=0, le=1)
    candidate_unsupported_rate: float = Field(ge=0, le=1)
    baseline_unsupported_rate: float = Field(ge=0, le=1)
    candidate_avg_request_ms: float = Field(ge=0)
    baseline_avg_request_ms: float = Field(ge=0)


class SelfModificationCanaryPolicyPayload(BaseModel):
    minimum_requests_per_cohort: int = Field(default=5, ge=5, le=100)
    target_requests_per_cohort: int = Field(default=20, ge=5, le=10_000)
    minimum_observation_seconds: int = Field(default=300, ge=0, le=604_800)
    minimum_verification_coverage: float = Field(default=0.8, ge=0.5, le=1)
    confidence_level: float = Field(default=0.95, ge=0.8, le=0.999)
    failure_rate_margin: float = Field(default=0.02, ge=0, le=0.25)
    unsupported_rate_margin: float = Field(default=0.02, ge=0, le=0.25)
    latency_multiplier: float = Field(default=1.5, ge=1, le=5)
    inconclusive_alert_seconds: int = Field(default=1_800, ge=0, le=2_592_000)
    alerts_enabled: bool = True


class SelfModificationRollbackPayload(BaseModel):
    reason: str = Field(default="operator", max_length=500)


_CANVAS_SOURCE_EXTENSIONS = {
    ".txt", ".md", ".csv", ".json", ".jsonl", ".yaml", ".yml",
    ".pdf", ".docx", ".xlsx", ".html", ".htm", ".xml",
}
_CANVAS_SOURCE_MAX_BYTES = 5 * 1024 * 1024


def _summary(record: dict) -> dict:
    return {
        "experiment_id": record["experiment_id"], "title": record["title"],
        "problem": record["problem"][:400], "domain": record.get("domain", "general"),
        "status": record["status"],
        "models": [item["model"] for item in record.get("participants", [])],
        "rounds": record.get("rounds", 1), "current_round": record.get("current_round", 0),
        "current_replica": record.get("current_replica", 0),
        "total_replicas": record.get("total_replicas", 1),
        "current_participant": record.get("current_participant", ""),
        "current_participants": record.get("current_participants", []),
        "topology": record.get("topology", {}),
        "calls_used": record.get("calls_used", 0), "max_calls": record.get("max_calls", 0),
        "dataset_count": len(record.get("datasets", [])),
        "contribution_count": len(record.get("contributions", [])),
        "created_at": record.get("created_at", ""), "updated_at": record.get("updated_at", ""),
        "error": record.get("error", ""),
        "run": record.get("run_envelope", {}),
    }


@router.get("/experiments/models")
async def experiment_models():
    from remy.config.settings import settings
    from remy.core.model_registry import list_registered_models

    models, seen = [], set()
    for item in list_registered_models():
        name = str(item.get("name") or "")
        if name and name not in seen and (item.get("has_key") or name.startswith("llamacpp:")):
            seen.add(name)
            models.append({"name": name, "provider": item.get("provider", ""), "connected": True})
    default = str(settings.SUMMARY_MODEL or "")
    if default and default not in seen:
        models.insert(0, {"name": default, "provider": "default", "connected": True})
    return {"models": models}


def _self_mod_lab():
    from remy.core.self_modification_lab import get_self_modification_lab

    return get_self_modification_lab()


def _active_project_id() -> str:
    from remy.core.microbrain import current_project_id

    return current_project_id()


@router.get("/experiments/self-modifications")
async def list_self_modification_proposals(status: str = "all", limit: int = 100):
    lab = _self_mod_lab()
    project_id = _active_project_id()
    policy = lab.get_canary_policy(project_id=project_id)
    return {
        "proposals": lab.list(
            project_id=project_id,
            status=status,
            limit=limit,
        ),
        "policy": policy,
        "constraints": {
            "target": "agent.guidance",
            "additive_only": True,
            "immutable": ["code", "base_prompt", "tools", "policy", "approval", "sandbox"],
            "min_eval_cases": 3,
            "canary_percent_range": [5, 25],
            "canary_min_requests_per_cohort": policy["minimum_requests_per_cohort"],
            "canary_target_requests_per_cohort": policy["target_requests_per_cohort"],
            "canary_min_observation_seconds": policy["minimum_observation_seconds"],
            "canary_min_verification_coverage": policy["minimum_verification_coverage"],
            "canary_confidence_level": policy["confidence_level"],
        },
    }


@router.put("/experiments/self-modifications/policy")
async def update_self_modification_canary_policy(
    payload: SelfModificationCanaryPolicyPayload,
):
    try:
        policy = _self_mod_lab().update_canary_policy(
            project_id=_active_project_id(),
            policy=payload.model_dump(),
        )
        return {"policy": policy}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/experiments/self-modifications")
async def create_self_modification_proposal(payload: SelfModificationProposalPayload):
    try:
        proposal = _self_mod_lab().create_proposal(
            project_id=_active_project_id(),
            **payload.model_dump(),
        )
        return {"proposal": proposal}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/experiments/self-modifications/{proposal_id}")
async def get_self_modification_proposal(proposal_id: str):
    try:
        return {
            "proposal": _self_mod_lab().get(
                project_id=_active_project_id(),
                proposal_id=proposal_id,
            )
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Self-modification proposal not found") from exc


@router.post("/experiments/self-modifications/{proposal_id}/evaluation")
async def evaluate_self_modification_proposal(
    proposal_id: str,
    payload: SelfModificationEvalPayload,
):
    try:
        proposal = _self_mod_lab().record_evaluation(
            project_id=_active_project_id(),
            proposal_id=proposal_id,
            matrix_id=payload.matrix_id,
        )
        return {"proposal": proposal}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Proposal or eval matrix not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/experiments/self-modifications/{proposal_id}/run-evaluation")
async def run_self_modification_evaluation(
    proposal_id: str,
    payload: SelfModificationEvalRunPayload,
):
    """Run sandbox replay with the exact immutable candidate overlay bound."""
    from remy.config.settings import settings
    from remy.core.trajectory_store import get_trajectory_store
    from remy.web.routes.trajectory_routes import _run_eval_matrix, _selected_eval_cases

    project_id = _active_project_id()
    lab = _self_mod_lab()
    trajectory_store = get_trajectory_store()
    try:
        proposal = lab.get(project_id=project_id, proposal_id=proposal_id)
        cases = _selected_eval_cases(
            trajectory_store=trajectory_store,
            project_id=project_id,
            case_ids=payload.case_ids,
        )
        with lab.evaluation_overlay(project_id=project_id, proposal_id=proposal_id):
            matrix = await _run_eval_matrix(
                project_id=project_id,
                trajectory_store=trajectory_store,
                cases=cases,
                name=(payload.name or f"Self-modification {proposal['candidate_hash'][:12]}")[:160],
                preferred_model=str(payload.preferred_model or settings.SUMMARY_MODEL or "")[:240],
                agent_version=f"self-mod:{proposal['candidate_hash']}",
            )
        proposal = lab.record_evaluation(
            project_id=project_id,
            proposal_id=proposal_id,
            matrix_id=matrix["matrix_id"],
        )
        return {"proposal": proposal, "matrix": matrix}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Proposal or eval case not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/experiments/self-modifications/{proposal_id}/approve")
async def approve_self_modification_proposal(
    proposal_id: str,
    payload: SelfModificationApprovalPayload,
):
    try:
        proposal = _self_mod_lab().approve(
            project_id=_active_project_id(),
            proposal_id=proposal_id,
            **payload.model_dump(),
        )
        return {"proposal": proposal}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Self-modification proposal not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/experiments/self-modifications/{proposal_id}/canary")
async def start_self_modification_canary(
    proposal_id: str,
    payload: SelfModificationCanaryPayload,
):
    try:
        proposal = _self_mod_lab().start_canary(
            project_id=_active_project_id(),
            proposal_id=proposal_id,
            **payload.model_dump(),
        )
        return {"proposal": proposal}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Self-modification proposal not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/experiments/self-modifications/{proposal_id}/canary-evaluation")
async def evaluate_self_modification_canary(
    proposal_id: str,
    payload: SelfModificationCanaryEvaluationPayload,
):
    try:
        proposal = _self_mod_lab().evaluate_canary(
            project_id=_active_project_id(),
            proposal_id=proposal_id,
            metrics=payload.model_dump(),
        )
        return {"proposal": proposal}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Self-modification proposal not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/experiments/self-modifications/{proposal_id}/canary-telemetry")
async def get_self_modification_canary_telemetry(proposal_id: str):
    """Preview aggregate production telemetry without changing lifecycle state."""
    from remy.core.trajectory_store import get_trajectory_store

    try:
        telemetry = _self_mod_lab().collect_canary_telemetry(
            project_id=_active_project_id(),
            proposal_id=proposal_id,
            trajectory_store=get_trajectory_store(),
        )
        return {"telemetry": telemetry}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Self-modification proposal not found") from exc


@router.post("/experiments/self-modifications/{proposal_id}/canary-telemetry/observe")
async def observe_self_modification_canary_telemetry(proposal_id: str):
    """Collect aggregate telemetry and enforce the gate when evidence is ready."""
    from remy.core.trajectory_store import get_trajectory_store

    try:
        return _self_mod_lab().observe_canary_telemetry(
            project_id=_active_project_id(),
            proposal_id=proposal_id,
            trajectory_store=get_trajectory_store(),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Self-modification proposal not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/experiments/self-modifications/{proposal_id}/promote")
async def promote_self_modification_proposal(
    proposal_id: str,
    payload: SelfModificationCanaryPayload,
):
    try:
        proposal = _self_mod_lab().promote(
            project_id=_active_project_id(),
            proposal_id=proposal_id,
            confirm_candidate_hash=payload.confirm_candidate_hash,
        )
        return {"proposal": proposal}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Self-modification proposal not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/experiments/self-modifications/{proposal_id}/rollback")
async def rollback_self_modification_proposal(
    proposal_id: str,
    payload: SelfModificationRollbackPayload,
):
    try:
        proposal = _self_mod_lab().rollback(
            project_id=_active_project_id(),
            proposal_id=proposal_id,
            reason=payload.reason,
        )
        return {"proposal": proposal}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Self-modification proposal not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/experiments")
async def list_experiments():
    return {"experiments": [_summary(record) for record in get_experiment_engine().store.list()]}


@router.get("/experiments/canvas/templates")
async def experiment_canvas_templates():
    from remy.core.experiment_canvas import canvas_templates

    return {"templates": canvas_templates()}


@router.get("/experiments/roles")
async def experiment_role_catalog():
    from remy.core.experiment_canvas import role_catalog

    return {"roles": role_catalog()}


@router.post("/experiments/canvas/template")
async def experiment_canvas_template(payload: CanvasTemplateRequest):
    from remy.core.experiment_canvas import build_template

    return {"canvas": build_template(payload.template_id, payload.models)}


@router.post("/experiments/canvas/source-file")
async def extract_canvas_source_file(file: UploadFile = File(...)):
    """Extract a bounded document for embedding in a not-yet-saved canvas."""
    from remy.config.settings import settings
    from remy.core.corpus_preprocessor import extract_clean_text

    filename = Path(file.filename or "source.txt").name
    suffix = Path(filename).suffix.lower()
    if suffix not in _CANVAS_SOURCE_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported document type. Allowed: {', '.join(sorted(_CANVAS_SOURCE_EXTENSIONS))}",
        )
    raw = await file.read(_CANVAS_SOURCE_MAX_BYTES + 1)
    if len(raw) > _CANVAS_SOURCE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Document is larger than 5 MB.")
    if not raw:
        raise HTTPException(status_code=422, detail="Document is empty.")

    temp_root = get_experiment_engine().store.root / ".canvas-imports"
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            path = Path(directory) / f"source{suffix}"
            path.write_bytes(raw)
            content, warnings = extract_clean_text(
                path, max_bytes_per_file=_CANVAS_SOURCE_MAX_BYTES, extractor_provider="built_in"
            )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read this document: {exc}") from exc
    content = str(content or "").strip()[:80_000]
    if not content:
        raise HTTPException(status_code=422, detail="No readable text was found in the document.")
    return {
        "name": filename[:200],
        "content": content,
        "characters": len(content),
        "warnings": list(warnings or []),
    }


@router.post("/experiments/canvas/validate")
async def validate_experiment_canvas(payload: CanvasPayload):
    from remy.core.experiment_canvas import compile_canvas, validate_canvas

    connected = {item["name"] for item in (await experiment_models())["models"]}
    errors = validate_canvas(payload.canvas, connected_models=connected)
    if errors:
        return {"valid": False, "errors": errors, "plan": None}
    return {"valid": True, "errors": [], "plan": compile_canvas(payload.canvas, connected_models=connected)}


@router.post("/experiments/canvas")
async def create_canvas_experiment(payload: CanvasPayload):
    from remy.core.experiment_canvas import compile_canvas

    engine = get_experiment_engine()
    connected = {item["name"] for item in (await experiment_models())["models"]}
    try:
        plan = compile_canvas(payload.canvas, connected_models=connected)
        record = engine.store.create(
            title=plan["title"], problem=plan["problem"], success_criteria=plan["success_criteria"],
            models=[item["model"] for item in plan["participants"]], rounds=plan["rounds"],
            max_calls=plan["max_calls"], domain=plan["domain"],
        )
        record = engine.store.mutate(record["experiment_id"], lambda item: item.update({
            "mode": "canvas", "canvas": payload.canvas, "experiment_plan": plan,
            "participants": plan["participants"],
            "total_replicas": int((plan.get("scenario") or {}).get("replicas") or 1),
            "events": [*item.get("events", []), {"at": item["created_at"], "type": "canvas_compiled", "message": f"Compiled {len(payload.canvas.get('nodes', []))} nodes"}],
        }))
        for dataset in plan.get("embedded_data", []):
            record = engine.store.add_dataset(
                record["experiment_id"], name=dataset["name"], content=dataset["content"], source="canvas"
            )
        return {"experiment": record, "plan": plan}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc).splitlines()) from exc


@router.post("/experiments")
async def create_experiment(payload: ExperimentCreate):
    engine = get_experiment_engine()
    connected = {item["name"] for item in (await experiment_models())["models"]}
    unknown = [model for model in payload.models if model not in connected]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Models are not connected: {', '.join(unknown)}")
    try:
        return {"experiment": engine.store.create(**payload.model_dump())}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.patch("/experiments/{experiment_id}")
async def update_experiment(experiment_id: str, payload: ExperimentUpdate):
    connected = {item["name"] for item in (await experiment_models())["models"]}
    unknown = [model for model in payload.models if model not in connected]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Models are not connected: {', '.join(unknown)}")
    try:
        record = get_experiment_engine().store.update_draft(experiment_id, **payload.model_dump())
        return {"experiment": record, "updated": True}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Experiment not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.put("/experiments/{experiment_id}/canvas")
async def update_canvas_experiment(experiment_id: str, payload: CanvasPayload):
    from remy.core.experiment_canvas import compile_canvas

    engine = get_experiment_engine()
    current = engine.store.get(experiment_id)
    if not current:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    if current.get("status") != "draft":
        raise HTTPException(status_code=409, detail="Only draft experiments can be edited.")
    if current.get("mode") != "canvas":
        raise HTTPException(status_code=409, detail="This is not a Canvas experiment.")
    connected = {item["name"] for item in (await experiment_models())["models"]}
    try:
        plan = compile_canvas(payload.canvas, connected_models=connected)
        record = engine.store.mutate(experiment_id, lambda item: item.update({
            "title": plan["title"], "problem": plan["problem"],
            "success_criteria": plan["success_criteria"], "domain": plan["domain"],
            "canvas": payload.canvas, "experiment_plan": plan, "participants": plan["participants"],
            "rounds": plan["rounds"], "max_calls": plan["max_calls"],
            "total_replicas": int((plan.get("scenario") or {}).get("replicas") or 1),
            "events": [*item.get("events", []), {"at": item["updated_at"], "type": "canvas_edited", "message": "Canvas draft updated"}],
        }))
        record = engine.store.replace_canvas_datasets(experiment_id, plan.get("embedded_data", []))
        return {"experiment": record, "plan": plan, "updated": True}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc).splitlines()) from exc


@router.delete("/experiments/{experiment_id}")
async def delete_experiment(experiment_id: str):
    engine = get_experiment_engine()
    if engine.is_active(experiment_id):
        raise HTTPException(status_code=409, detail="Stop the running experiment before deleting it.")
    try:
        return engine.store.delete_recoverably(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Experiment not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    engine = get_experiment_engine()
    record = engine.store.get(experiment_id)
    if not record:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    return {"experiment": record, "active": engine.is_active(experiment_id)}


@router.post("/experiments/{experiment_id}/data")
async def add_experiment_data(experiment_id: str, payload: ExperimentData):
    try:
        record = get_experiment_engine().store.add_dataset(experiment_id, name=payload.name, content=payload.content)
        return {"experiment": record}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Experiment not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/experiments/{experiment_id}/files")
async def add_experiment_file(experiment_id: str, file: UploadFile = File(...)):
    allowed = {".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".log"}
    if Path(file.filename or "").suffix.lower() not in allowed:
        raise HTTPException(status_code=415, detail="Use a text, Markdown, CSV, JSON, YAML, or log file.")
    raw = await file.read(1_000_001)
    if len(raw) > 1_000_000:
        raise HTTPException(status_code=413, detail="File is larger than 1 MB.")
    try:
        record = get_experiment_engine().store.add_dataset(
            experiment_id, name=file.filename or "Uploaded data", content=raw.decode("utf-8", errors="replace")
        )
        return {"experiment": record}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Experiment not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/experiments/{experiment_id}/start")
async def start_experiment(experiment_id: str):
    try:
        return {"experiment": get_experiment_engine().start(experiment_id), "started": True}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Experiment not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/experiments/{experiment_id}/continue")
async def continue_experiment(experiment_id: str, payload: ExperimentContinue):
    try:
        record = get_experiment_engine().continue_experiment(
            experiment_id, question=payload.question, rounds=payload.rounds
        )
        return {"experiment": record, "started": True, "continued": True}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Experiment not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/experiments/{experiment_id}/interventions")
async def add_scenario_intervention(experiment_id: str, payload: ScenarioIntervention):
    try:
        record = get_experiment_engine().add_intervention(
            experiment_id, content=payload.content, round_no=payload.round
        )
        return {"experiment": record, "scheduled": True}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Experiment not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/experiments/{experiment_id}/stop")
async def stop_experiment(experiment_id: str):
    try:
        return {"experiment": get_experiment_engine().stop(experiment_id), "stop_requested": True}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Experiment not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/experiments/{experiment_id}/pause")
async def pause_experiment(experiment_id: str):
    try:
        return {
            "experiment": get_experiment_engine().pause(experiment_id),
            "pause_requested": True,
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Experiment not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/experiments/{experiment_id}/resume")
async def resume_experiment(experiment_id: str):
    try:
        return {
            "experiment": get_experiment_engine().resume(experiment_id),
            "resumed": True,
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Experiment not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/experiments/{experiment_id}/synthesis/approve")
async def approve_experiment_synthesis(experiment_id: str):
    try:
        return {
            "experiment": get_experiment_engine().decide_synthesis(
                experiment_id,
                approved=True,
            ),
            "approved": True,
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Experiment not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/experiments/{experiment_id}/synthesis/reject")
async def reject_experiment_synthesis(experiment_id: str):
    try:
        return {
            "experiment": get_experiment_engine().decide_synthesis(
                experiment_id,
                approved=False,
            ),
            "approved": False,
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Experiment not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
