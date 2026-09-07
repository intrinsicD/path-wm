"""The new paddle experiment CLI; existing LeWM commands remain unchanged."""
from .paddle.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
