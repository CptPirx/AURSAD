import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="UR-data",
    version="0.1",
    author="Błażej Leporowski",
    author_email="bleporowski@outlook.com",
    description="Accompanying library to UR screwdriver dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Windows",
    ],
    python_requires='>=3.6',
)