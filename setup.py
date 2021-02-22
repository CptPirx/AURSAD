import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="aursad",
    version="0.1.7",
    author="Błażej Leporowski",
    author_email="bleporowski@outlook.com",
    description="Accompanying library to the AURSAD dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CptPirx/robo-package",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows :: Windows 10",
    ],
    python_requires='>=3.6',
    install_requires=[
        'tqdm',
        'sklearn',
        'numpy',
        'pandas',
        'random',
        'tensorflow'
    ]
)