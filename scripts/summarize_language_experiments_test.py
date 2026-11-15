from scripts.summarize_language_experiments import collect


def test_final_checkpoint_ledger_reads_raw_counts_and_preserves_missing_results():
    rows = {row["run"]: row for row in collect()}
    assert rows["canonical_30k"]["para"]["successes"] == 21
    assert rows["teacher_consistency_lam1_30k"]["para"]["successes"] == 57
    assert rows["teacher_consistency_lam10_30k"]["plus"]["successes"] == 234
    assert rows["consistency_lam01_30k"]["goal"]["successes"] == 84
    assert rows["consistency_lam01_30k"]["plus"] is None
    assert rows["consistency_lam10_30k"]["plus"] is None
    assert rows["teacher_consistency_lam1_strict_lora_30k"]["plus"]["episodes"] == 410
