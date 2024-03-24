import setuptools
from setuptools import find_packages
import pip
import sys

setuptools.setup(
    name="optuna",
    packages=find_packages(),
    version="0.0.1",
    description="Predict the demand side price",
)
