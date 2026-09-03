"""Repository-level pytest CLI options needed before test-path discovery."""


def pytest_addoption(parser):
    parser.addoption(
        "--run-dir",
        default="runs/dev/first_slice/0",
        help="checkpoint run directory used by threshold conformance tests",
    )
