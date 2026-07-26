from pathlib import Path

from run_pitch_bias_suites import replay_command


def test_replay_changes_only_method_and_output_scope():
    config = {
        "data_dir": "/data",
        "checkpoint": "/model.pt",
        "cache_dir": "/cache",
        "device": "cpu",
        "protocols": ["chronological", "interleaved"],
        "calibration_sizes": [10, 20],
        "repeats": 10,
        "gate_strategy": "reliability",
        "seed": 42,
        "split_json": "/split.json",
        "fold_index": 2,
        "sids": "",
    }
    command = replay_command(config, Path("/new-output"))
    assert command[command.index("--methods") + 1] == "pitch_bias"
    assert command[command.index("--output-dir") + 1] == "/new-output"
    assert command[command.index("--checkpoint") + 1] == "/model.pt"
    assert command[command.index("--seed") + 1] == "42"
    assert command[command.index("--split-json") + 1] == "/split.json"
