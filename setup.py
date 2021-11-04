import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


setup(
    name="gigalens",
    version="0.1.0",
    # url="https://github.com/kragniz/cookiecutter-pypackage-minimal",
    license='MIT',

    author="Andi Gu",
    author_email="andi.gu@berkeley.edu",

    description="Fast strong gravitational lens modeling",
    long_description=read("README.rst"),

    packages=find_packages(exclude=('tests',)),

    install_requires=[
        "tensorflow==2.6.0",
        "tensorflow-probability==0.14.1",
        "lenstronomy >= 1.9.1",
        "matplotlib >= 3.2.2",
        "scikit-image>=0.16.2",
        "cosmohammer >= 0.6.1",
        "schwimmbad >= 0.3.2",
        "dynesty >= 1.1",
        "corner >= 2.2.1"
    ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
)
