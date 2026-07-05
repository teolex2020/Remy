"""
Cycle execution runtime for Remy v3.

Owns the full mission cycle pipeline so ChiefAgent.run_cycle becomes a thin
wrapper over the runtime graph.
"""

from __future__ import annotations


class CycleExecutionRuntime:
    """Execute one full mission cycle through the extracted runtimes."""

    async def run(self, chief, *, mission, cycle_num):
        cycle_start = chief.timing_runtime.start_cycle()
        result = chief.cycle_result_runtime.initial(mission_id=mission.id)

        cycle_prep = chief.cycle_runtime.prepare(mission)
        if not cycle_prep.proceed:
            chief.cycle_result_runtime.apply_cycle_prep(result, cycle_prep=cycle_prep)
            chief.checkpoint_runtime.early_exit(chief)
            return result

        goal = cycle_prep.goal
        current_task = cycle_prep.task
        plan = cycle_prep.plan
        step = cycle_prep.step
        memory_context = cycle_prep.memory_context or []

        chief.cycle_result_runtime.apply_context(
            result,
            goal=goal,
            memory_context=memory_context,
        )

        gate = chief.execution_gate.prepare(
            mission=mission,
            goal=goal,
            task=current_task,
            plan=plan,
            step=step,
            memory_context=memory_context,
        )
        if not gate.proceed:
            chief.cycle_result_runtime.apply_gate(result, gate=gate)
            chief.checkpoint_runtime.early_exit(chief)
            return result

        specialist = gate.specialist
        chief.cycle_result_runtime.apply_specialist(result, specialist=specialist)

        exec_result = await chief.execution_runtime.execute(agent_ctx=gate.agent_ctx)
        exec_result = chief.factuality_runtime.apply(exec_result)
        duration_ms = chief.timing_runtime.elapsed_ms(cycle_start)

        chief.cycle_result_runtime.apply_execution(result, exec_result=exec_result)
        chief.cost_runtime.apply(mission=mission, exec_result=exec_result)

        eval_result = chief.evaluation_runtime.evaluate(
            exec_result=exec_result,
            goal=goal,
            task=current_task,
            specialist_id=specialist.id,
        )
        chief.cycle_result_runtime.apply_evaluation(
            result,
            eval_result=eval_result,
            step_id=step.id,
        )
        if getattr(chief, "evidence_debt_runtime", None) is not None:
            chief.evidence_debt_runtime.resolve_after_evaluation(
                task=current_task,
                mission=mission,
                goal=goal,
                step=step,
                specialist=specialist,
                exec_result=exec_result,
                eval_result=eval_result,
            )

        outcome = chief.outcome_runtime.apply(
            mission=mission,
            goal=goal,
            task=current_task,
            plan=plan,
            step=step,
            exec_result=exec_result,
            eval_result=eval_result,
        )
        chief.cycle_result_runtime.apply_outcome(result, outcome=outcome)

        chief.post_cycle_runtime.finalize(
            chief,
            cycle_num=cycle_num,
            mission=mission,
            goal=goal,
            task=current_task,
            plan=plan,
            step=step,
            specialist=specialist,
            exec_result=exec_result,
            eval_result=eval_result,
            decision=result.decision,
            memory_assisted=result.memory_context_used,
            duration_ms=duration_ms,
        )
        return result
