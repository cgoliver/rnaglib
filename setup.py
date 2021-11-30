import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = ["torch",
                "dgl>=0.6",
                'networkx',
                "numpy",
                "seaborn",
                "sklearn",
                "tqdm",
                "biopython"
                ]

setuptools.setup(
    name="rnaglib",
    version="1.0.1",
    author="Vincent Mallet, Carlos Oliver, Jonathan Broadbent, William L. Hamilton and Jérome Waldispuhl",
    author_email="vincent.mallet96@gmail.com",
    description="RNAglib: Tools for learning on the structure of RNA using 2.5D graph representations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://jwgitlab.cs.mcgill.ca/cgoliver/rnaglib",
    packages=setuptools.find_packages(),
    package_data={'rnaglib': ['data_loading/graph_index_NR.json']},
    # include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # python_requires='>=3.5',
    scripts=['rnaglib/bin/rnaglib_first',
             'rnaglib/bin/rnaglib_second',
             'rnaglib/bin/rnaglib_prepare_data',
             'rnaglib/bin/rnaglib_download']
)
