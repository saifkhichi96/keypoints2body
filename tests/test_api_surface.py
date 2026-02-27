from keypoints2body import optimize_params_frame, optimize_params_sequence


def test_api_exports_exist():
    assert callable(optimize_params_frame)
    assert callable(optimize_params_sequence)
