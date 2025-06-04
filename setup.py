from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="hlda",
    version="0.4",
    author="Joe Wandy",
    author_email="joe.wandy@glasgow.ac.uk",
    description = 'Gibbs sampler for the Hierarchical Latent Dirichlet Allocation topic model. This is based on the hLDA implementation from Mallet, having a fixed depth on the nCRP tree.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url = 'https://github.com/joewandy/hlda', # use the URL to the github repo
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
    packages=find_packages(),
    install_requires=['numpy'],
)
