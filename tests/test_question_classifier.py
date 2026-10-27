from contextual_rl_agentic_rag.rl.question_classifier import classify_question


def test_factual():
    assert classify_question("What is retrieval augmented generation?") == "factual"


def test_analytical():
    assert classify_question("Compare the average revenue grouped by region") == "analytical"


def test_relational():
    # NOTE: patterns are word-boundary stems, so "hierarchy"/"between" match but
    # inflections like "connected" do not (a known classifier limitation).
    assert classify_question("Describe the hierarchy between the two teams") == "relational"


def test_procedural():
    assert classify_question("How to configure the vector store step by step?") == "procedural"


def test_visual():
    assert classify_question("Describe the chart in the diagram") == "visual"


def test_vague_short_query():
    assert classify_question("tell me") == "vague"


def test_default_falls_back_to_factual():
    assert classify_question("the quick brown fox jumps over the lazy dog again") == "factual"
