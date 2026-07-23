import importlib.util, pathlib, pytest
spec = importlib.util.spec_from_file_location(
    "build_flags", pathlib.Path(__file__).parent.parent / "scripts" / "build_flags.py")
bf = importlib.util.module_from_spec(spec); spec.loader.exec_module(bf)


def test_all_backends_enable_rpc():
    for b in ("cuda", "metal", "vulkan", "cpu"):
        assert "-DGGML_RPC=ON" in bf.cmake_flags(b)


def test_backend_specific_flag():
    assert "-DGGML_CUDA=ON" in bf.cmake_flags("cuda")
    assert "-DGGML_METAL=ON" in bf.cmake_flags("metal")
    assert "-DGGML_VULKAN=ON" in bf.cmake_flags("vulkan")
    assert not any("CUDA" in f or "METAL" in f or "VULKAN" in f for f in bf.cmake_flags("cpu"))


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        bf.cmake_flags("tpu")
