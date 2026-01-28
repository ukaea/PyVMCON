import pytest  # noqa: D100


@pytest.fixture(params=[True, False], ids=["slack_constraints", "no_slack_constraints"])
def matrix_over_allow_slack_constraints(request):
    """A fixture to matrix a test over [True, False].

    This is used in the context of testing pyvmcon.solve both with and without slack
    constraint bounds.
    """
    return request.param
