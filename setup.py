from setuptools import setup, find_packages

setup(
    name='almiky',
    version='0.1',
    description='Python library for data hiding in images',
    url='https://gitlab.udg.co.cu/Investigacion/almiky',
    author='Yenner J. Diaz-Nuñez, Anier Soria-Lorente, Ernesto Avila-Domenech',
    author_email='yennerdiaz@gmail.com',
    license='LICENSE.txt',
    packages=find_packages(),
    install_requires=[
        'numpy==1.24.3',
        'scipy==1.10.1',
        'mpmath==1.3.0',
        'imageio==2.28.1',
    ],
    zip_safe=False
)
