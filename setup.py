from setuptools import setup, find_packages

setup(name='ssBind',
      version='0.1.0',
      description='ssBind - Substructure-based alternative BINDing modes generator for protein-ligand systems',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Natural Language :: English',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'Topic :: Scientific/Engineering :: Chemistry'
      ],
      url='https://github.com/suleymanselim/ssBind',
      author='Suleyman Selim Cinaroglu',
      author_email='scinarog@uci.edu',
      license='MIT',
      keywords=["molecular modeling", "drug design",
            "docking", "protein-ligand"],
      packages=find_packages(),
      scripts=['ssBind/run_ssBind.py'],
      install_requires=[
          'openbabel',
          'numpy',
          'rdkit',
          'pandas',
          'MDAnalysis',
          'rpy2'
      ],
      zip_safe=False)
