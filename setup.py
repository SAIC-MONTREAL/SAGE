"""Setups up the SAGE module."""

from setuptools import find_packages, setup


def get_description():
    """Gets the description from the readme."""
    with open("README.md") as fh:
        long_description = ""
        header_count = 0

        for line in fh:
            if line.startswith("##"):
                header_count += 1

            if header_count < 2:
                long_description += line
            else:
                break

    return header_count, long_description


def get_version():
    """Gets the version."""
    path = "sage/__init__.py"
    with open(path) as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


version = get_version()
header_count, long_description = get_description()

setup(
    name="sage",
    version=version,
    author="SAIC Montreal",
    description="Smart Home Agent with Grounded Execution",
    license="CC-BY-NV 3.0",
    licence_files=("LICENCE",),
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=["LLMs", "Autonomous Agents", "IoT", "Multi-modal"],
    python_requires=">=3.10,<3.11",
    packages=find_packages(exclude=["scripts"]),
    include_package_data=True,
    install_requires=[
        "numpy==1.24.2",
        "pyaml",
        "tqdm==4.65.0",
        "pandas==1.5.3",
        "seaborn==0.12.2",
        "pre-commit",
    ],
)
