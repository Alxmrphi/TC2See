from setuptools import setup

setup(name='fmri_processing',
      version="0.1",
      packages=['fmri_processing'],
      description='Process and analyse fMRI recordings',
      author='Richard Gerum',
      author_email='richard.gerum@fau.de',
      install_requires=[
          'numpy',
          'scipy',
          'tensorflow',
          'pandas',
          'nilearn',
      ],
)
