from remy.core.tool_contracts import (
    REQUIRED_CONTRACT_SECTIONS,
    ROUTING_BENCHMARK_CASES,
    TOOL_CONTRACTS,
    run_tool_routing_benchmark,
    validate_tool_contracts,
)


def _declarations_by_name(declarations):
    return {item.name: item for item in declarations}


def test_tool_contract_registry_is_complete_and_well_formed():
    assert not validate_tool_contracts()
    assert len(TOOL_CONTRACTS) >= 10
    for contract in TOOL_CONTRACTS.values():
        rendered = contract.render()
        assert all(section in rendered for section in REQUIRED_CONTRACT_SECTIONS)


def test_runtime_declarations_receive_the_same_compiled_contracts():
    from remy.core.brain_tools import BRAIN_TOOLS as runtime_declarations
    from remy.core.tool_declarations import BRAIN_TOOLS as modular_declarations

    runtime = _declarations_by_name(runtime_declarations)
    modular = _declarations_by_name(modular_declarations)
    for name, contract in TOOL_CONTRACTS.items():
        assert runtime[name].description == contract.render()
        assert modular[name].description == contract.render()


def test_routing_benchmark_reports_accuracy_and_forbidden_gravity():
    perfect = run_tool_routing_benchmark(lambda case: case.expected_tool)
    assert perfect["passed"] is True
    assert perfect["accuracy"] == 1.0
    assert perfect["safe_routing_rate"] == 1.0

    shell_gravity = run_tool_routing_benchmark(lambda _case: "shell_exec")
    assert shell_gravity["passed"] is False
    assert shell_gravity["accuracy"] < 0.2
    assert shell_gravity["forbidden_selections"] > 0


def test_routing_benchmark_covers_every_registered_contract():
    expected = {case.expected_tool for case in ROUTING_BENCHMARK_CASES}
    assert expected == set(TOOL_CONTRACTS)
