import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="Funct",
    version="0.9.2",
    author="Lauri Tuominen",
    author_email="lauri@port6.io",
    description="Functional Python Sequence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lauriat/funct",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
