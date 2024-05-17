from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='primo',
    version='1.0.0',
    author='Atharva Mete',
    author_email='amete7@gatech.edu',
    description='Code for PRIMO: Towards Robot Learning with Motion Primitives',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)