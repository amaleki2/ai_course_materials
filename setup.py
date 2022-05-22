from setuptools import setup
from setuptools import find_packages

setup(
    name='ai_course',
    author='Amir Maleki',
    description="This package is part of Stanford AI course",
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "graphviz",
        "sklearn",
        "Pillow"
    ],
)
