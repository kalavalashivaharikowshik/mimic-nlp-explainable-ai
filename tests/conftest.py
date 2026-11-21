import os

# Never try to download the real model during tests.
os.environ.setdefault("LOAD_MODEL_ON_STARTUP", "false")

import pytest  # noqa: E402

from backend.predict import MortalityPredictor  # noqa: E402
from tests.fakes import make_artifacts, make_onnx_artifacts  # noqa: E402

STABLE = [45, 0, 76, 122, 78, 93, 98, 0.9, 104, 14.1]
CRITICAL = [78, 1, 122, 84, 46, 58, 87, 3.1, 248, 8.2]


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    return make_artifacts(tmp_path_factory.mktemp("tok"))


@pytest.fixture(scope="session")
def onnx_bundle(tmp_path_factory):
    """(OnnxArtifacts, TorchArtifacts) of the same tiny model."""
    return make_onnx_artifacts(tmp_path_factory.mktemp("onnx"))


@pytest.fixture(scope="session")
def predictor(onnx_bundle):
    return MortalityPredictor(onnx_bundle[0])
