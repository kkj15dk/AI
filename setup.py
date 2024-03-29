from setuptools import setup, find_packages

# Read requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="your_project_name",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,  # Use the list of requirements read from requirements.txt

    # Additional metadata
    author="Kasper Krunderup Jakobsen",
    author_email="kasperkrunderup@gmail.com",
    description="A project for analyzing biosynthetic gene clusters (BGCs) in genomes, and for predicting the chemical structures and activities of the compounds they encode.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
)