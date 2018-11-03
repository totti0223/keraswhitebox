import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="keraswhitebox",
    version="0.1",
    author="Yosuke Toda",
    author_email="tyosuke@aquaseerser.com",
    description="ToolBox for visualizing neural networks built with Keras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)