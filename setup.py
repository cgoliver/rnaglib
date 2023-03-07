import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = open("requirements.txt", "r").readlines()

setuptools.setup(
    name="rnaglib",
    version="2.0.0",
    author="Vincent Mallet, Carlos Oliver, Jonathan Broadbent, William L. Hamilton and Jérome Waldispuhl",
    author_email="vincent.mallet96@gmail.com",
    description="RNAglib: Tools for learning on the structure of RNA using 2.5D geometric representations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://rnaglib.readthedocs.io/en/latest/index.html",
    packages=setuptools.find_packages(),
    package_data={'rnaglib': ['data_loading/graph_index_NR.json']},
    # include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    scripts=['rnaglib/examples/rnaglib_first',
             'rnaglib/examples/rnaglib_second',
             'rnaglib/bin/rnaglib_prepare_data',
             'rnaglib/bin/rnaglib_tokenize',
             'rnaglib/bin/rnaglib_download']
)
