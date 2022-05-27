import io
import os
import re

from setuptools import find_namespace_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())

jax_deps = ["jax==0.2.24", "optax==0.0.9", "objax==1.4.0"]
test_deps = [
    "coverage",
    "pytest",
    *jax_deps
]

extras = {
    "test": test_deps,
    "jax": jax_deps,
}

setup(
    name="gigalens",
    version="0.1.8",
    license="MIT",
    author="Andi Gu",
    author_email="andi.gu@berkeley.edu",
    description="Fast strong gravitational lens modeling",
    long_description=read("README.rst"),
    long_description_content_type='text/x-rst',
    packages=find_namespace_packages(where='src'),
    package_dir={"": "src"},
    package_data={'': ['*.npy']},
    # include_package_data=True,
    install_requires=[
        "tensorflow>=2.6.0",
        "tensorflow-probability>=0.15.0",
        "lenstronomy==1.9.3",
        "scikit-image==0.18.2",
        # The following packages are required by lenstronomy but are not automatically installed
        "cosmohammer==0.6.1",
        "schwimmbad==0.3.2",
        "dynesty==1.1",
        "corner==2.2.1",
        "mpmath==1.2.1",
        "tqdm==4.62.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
    ],
    tests_require=test_deps,
    extras_require=extras,
)
