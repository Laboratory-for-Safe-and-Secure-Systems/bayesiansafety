if __name__ == "__main__":
    import sys
    import pytest


    package_name = "bayesiansafety"
    pytest_args = [
        "--cov-config=.coveragerc",
        "--cov-report=html:./reports",
        "--cov=bayesiansafety/synthesis",
        "--cov=bayesiansafety/faulttree",
        "--cov=bayesiansafety/eventtree",
        "--cov=bayesiansafety/bowtie",
        "--cov=bayesiansafety/core",
        "--cov=bayesiansafety/utils",
        "--verbose",
        "--junitxml=./reports/junit_report.xml",
        "--cov-branch",
        "--new-first",
    ]
    pytest_args = pytest_args + sys.argv[1:]
    sys.exit(pytest.main(pytest_args))