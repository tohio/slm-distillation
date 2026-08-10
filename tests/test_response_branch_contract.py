from distill.artifacts.handoff import load_artifact_config
from distill.utils.config import (
    load_dpo_config,
    load_eval_config,
    load_export_config,
    load_response_distill_config,
)


def test_full_response_branch_uses_one_method_based_identity() -> None:
    response = load_response_distill_config("configs/response_distill.yaml")
    dpo = load_dpo_config("configs/dpo.yaml")
    evaluation = load_eval_config("configs/eval.yaml")
    export = load_export_config("configs/export.yaml")
    artifacts = load_artifact_config("configs/artifacts.yaml")

    assert response.output.model_name == "smollm2-135m-response-distilled"
    assert dpo.source.model_name == response.output.model_name
    assert dpo.source.checkpoint_path == response.output.final_checkpoint_dir
    assert dpo.source.tokenizer_path == response.output.final_checkpoint_dir
    assert dpo.output.model_name == response.output.model_name
    assert evaluation.models[1].model_name_or_path == (
        response.output.final_checkpoint_dir
    )
    assert evaluation.models[2].model_name_or_path == (
        dpo.output.final_checkpoint_dir
    )
    assert export.model.model_name == response.output.model_name
    assert export.model.checkpoint_path == dpo.output.final_checkpoint_dir
    assert export.model.tokenizer_path == dpo.output.final_checkpoint_dir
    assert export.model_card.eval_results_path == evaluation.output.results_path
    assert artifacts.run_name == response.output.model_name
    assert f"{dpo.output.final_checkpoint_dir}/*" in artifacts.include


def test_smoke_response_branch_preserves_isolated_handoffs() -> None:
    response = load_response_distill_config(
        "configs/response_distill_smoke.yaml"
    )
    dpo = load_dpo_config("configs/dpo_smoke.yaml")
    evaluation = load_eval_config("configs/eval_smoke.yaml")

    assert response.output.model_name == (
        "smollm2-135m-response-distilled-smoke"
    )
    assert dpo.source.model_name == response.output.model_name
    assert dpo.source.checkpoint_path == response.output.final_checkpoint_dir
    assert dpo.output.model_name == response.output.model_name
    assert evaluation.models[1].model_name_or_path == (
        response.output.final_checkpoint_dir
    )
    assert evaluation.models[2].model_name_or_path == (
        dpo.output.final_checkpoint_dir
    )
    assert response.output.final_checkpoint_dir.startswith("runs/smoke/")
    assert dpo.output.final_checkpoint_dir.startswith("runs/smoke/")
