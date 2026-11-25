"""PyInstaller entrypoint wrapper ensuring package-qualified imports."""

from dips.main import main

if __name__ == "__main__":
    raise SystemExit(main())
