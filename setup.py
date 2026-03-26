from setuptools import setup, find_packages

setup(name='fw4pde',
      version='1.0.2',
      description='Frank--Wolfe algorithms for PDE-constrained optimization',
      author="Johannes Milz",
      author_email='johannes.milz@gatech.edu',
      url='https://github.com/milzj/FW4PDE',
  package_dir={"": "src"},
  packages=find_packages(where="src"),
      install_requires=[
        "numpy",
        "scipy"
      ]
)
