from setuptools import setup, find_packages

setup(name='fw4pde',
      version='1.0.2',
      description='Frank--Wolfe algorithms for PDE-constrained optimization',
      author="Johannes Milz",
      author_email='johannes.milz@gatech.edu',
      url='https://github.com/milzj/FW4PDE',
      packages=find_packages(),
      install_requires=[
        "numpy",
        "scipy"
      ]
)
