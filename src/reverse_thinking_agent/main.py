from importlib.metadata import PackageNotFoundError, version


def _package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not installed"


def main() -> None:
    print("reverse-thinking-agent")
    print(f"TempoBench dependency: {_package_version('tempobench')}")


if __name__ == "__main__":
    main()
