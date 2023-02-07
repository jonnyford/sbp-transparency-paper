"""
Build and install the package.
"""
from setuptools import setup, find_packages

NAME = 'sbp_modelling'
FULLNAME = NAME
AUTHOR = "Jonathan Ford"
AUTHOR_EMAIL = 'jford@ogs.it'
LICENSE = ""
URL = ""
DESCRIPTION = ""
KEYWORDS = ''
LONG_DESCRIPTION = DESCRIPTION

VERSION = '0.01'

PACKAGES = find_packages(exclude=['tests', 'notebooks', 'figures'])
SCRIPTS = []

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
]
PLATFORMS = "Any"
INSTALL_REQUIRES = [
]

if __name__ == '__main__':
    setup(name=NAME,
          fullname=FULLNAME,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          version=VERSION,
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          license=LICENSE,
          url=URL,
          platforms=PLATFORMS,
          scripts=SCRIPTS,
          packages=PACKAGES,
          classifiers=CLASSIFIERS,
          keywords=KEYWORDS,
          install_requires=INSTALL_REQUIRES)
