import os
import setuptools

_here = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="yelp_analysis",
    version='0.0.2',
    author="Ian Buttimer",
    author_email="author@example.com",
    description="Yelp Open Dataset Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ib-da-ncirl/yelp_analysis",
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=[
      'pandas>=1.0.5',
      'dask>=2.19.0',
      'pillow>=7.2.0',
      'yaml>=0.2.5',
      'pyyaml>=5.3.1',
      'setuptools>=47.3.1',
      'tensorflow>=2.2.0',
      'tensorflow-gpu>=2.2.0',
      'numpy>=1.18.5',
      'matplotlib>=3.2.2'
      'keras>=2.4.3',
      'scikit-learn>=0.23.1',
    ],
    dependency_links=[
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
