# tests/test_judge_validation.py
def test_deepseek_judge_constructs_without_network():
    from judge.deepseek_judge import DeepSeekJudge
    j = DeepSeekJudge(api_key="dummy")  # no call made -> no network
    assert j.get_model_name().startswith("deepseek")
