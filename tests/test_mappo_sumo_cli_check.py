from scripts import collect_mappo_sumo, train_mappo_sumo


def forbidden(*args, **kwargs):
    raise AssertionError("Check mode must not collect, initialize or train a policy")


def test_training_check_validates_without_creating_output_or_optimizer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_mappo_sumo, "MAPPOTrainer", forbidden)
    monkeypatch.setattr(train_mappo_sumo, "rollout_mobile_session", forbidden)
    monkeypatch.setattr(train_mappo_sumo, "load_dataset", lambda _: (
        dict(manifest_sha256="validated", seed_splits=dict(train=[1], validation=[2], test=[3])),
        {},
    ))
    train_mappo_sumo.main(["--dataset", str(tmp_path / "dataset"), "--check"])
    assert list(tmp_path.iterdir()) == []


def test_collection_check_does_not_launch_sumo_or_create_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    net = tmp_path / "fixture.net.xml"
    net.write_text("fixture: check mode does not parse the road network")
    monkeypatch.setattr(collect_mappo_sumo.shutil, "which", lambda _: "/fake/sumo")
    monkeypatch.setattr(collect_mappo_sumo, "collect_trace", forbidden)
    collect_mappo_sumo.main([
        "--sumo-net", str(net), "--train-seeds", "1", "--validation-seeds", "2",
        "--test-seeds", "3", "--check",
    ])
    assert list(tmp_path.iterdir()) == [net]
