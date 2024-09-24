import setuptools
import os
import re

with open("README.md", "r") as fh:
    long_description = fh.read()


with open(os.path.join("MeanStars", "__init__.py"), "r") as f:
    version_file = f.read()

version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)

if version_match:
    version_string = version_match.group(1)
else:
    raise RuntimeError("Unable to find version string.")

descstr = ("Automated property interpolation and color calculations for "
           "main sequence stars")

setuptools.setup(
    name="MeanStars",
    version=version_string,
    author="Dmitry Savransky",
    author_email="ds264@cornell.edu",
    description=descstr,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dsavransky/MeanStars",
    packages=["MeanStars"],
    package_data={"MeanStars": ["EEM_dwarf_UBVIJHK_colors_Teff.txt"]},
    install_requires=["scipy", "numpy", "astropy"],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
