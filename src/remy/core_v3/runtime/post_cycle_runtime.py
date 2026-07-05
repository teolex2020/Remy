"""
Post-cycle runtime for Remy v3.

Coordinates normalized after-cycle side effects so the chief agent does not
manually orchestrate recording, learning, and checkpoint hooks inline.
"""

from __future__ import annotations


class PostCycleRuntime:
    """Apply post-cycle hooks in a deterministic order."""

    def __init__(self, recording_runtime, checkpoint_runtime, learning_runtime=None):
        self.recording_runtime = recording_runtime
        self.checkpoint_runtime = checkpoint_runtime
        self.learning_runtime = learning_runtime

    def finalize(
        self,
        chief,
        *,
        cycle_num: int,
        mission,
        goal,
        task,
        plan,
        step,
        specialist,
        exec_result,
        eval_result,
        decision: str,
        memory_assisted: bool,
        duration_ms: int,
    ) -> None:
        self.recording_runtime.record_cycle(
            cycle_num=cycle_num,
            mission=mission,
            goal=goal,
            task=task,
            plan=plan,
            step=step,
            specialist=specialist,
            exec_result=exec_result,
            eval_result=eval_result,
            decision=decision,
            memory_assisted=memory_assisted,
            duration_ms=duration_ms,
        )
        if self.learning_runtime is not None:
            self.learning_runtime.observe_cycle(
                mission=mission,
                goal=goal,
                task=task,
                step=step,
                specialist=specialist,
                exec_result=exec_result,
                eval_result=eval_result,
                decision=decision,
            )
        self.checkpoint_runtime.complete_cycle(chief, plan=plan)
