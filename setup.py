import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="Func",
    version="0.1",
    author="Lauri Tuominen",
    author_email="lauri@port6.io",
    description="Functional Python Sequence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lauriat/func",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
