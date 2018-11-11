import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MeanStars",
    version="1.0.1",
    author="Dmitry Savransky",
    author_email="ds264@cornell.edu",
    description="Automated property interpolation and color calculations for main sequence stars",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dsavransky/MeanStars",
    packages=['MeanStars'],
    package_data={'MeanStars': ['EEM_dwarf_UBVIJHK_colors_Teff.txt']},
    install_requires=[
          'scipy',
          'numpy',
          'astropy'
    ],
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
