from src.core.query_rewrite import expand_query


def test_expand_query_adds_policy_number_synonyms():
    expanded = expand_query("What is the policy number?")
    assert "policy #" in expanded
    assert "policy no" in expanded
    assert expanded.startswith("What is the policy number?")


def test_expand_query_handles_total_premium():
    expanded = expand_query("Show the estimated total premium")
    assert "total premium" in expanded.lower()
    assert "premium overall" in expanded.lower()


def test_expand_query_no_change_when_not_triggered():
    query = "Tell me about coverage limits"
    assert expand_query(query) == query
