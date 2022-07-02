from setuptools import setup, find_packages

install_requires = [
    'attrdict==2.0.1',
    'commonmark==0.9.1',
    'grpcio==1.41.0',
    'numpy==1.23.0',
    'opencv-python==4.6.0.66',
    'protobuf==3.19.4',
    'Pygments==2.12.0',
    'python-rapidjson==1.6',
    'rich==12.4.4',
    'six==1.16.0',
    'tritonclient==2.22.4',
]

setup(
    name='clients',
    version='0.1.0',

    python_requires='>=3.8',

    packages=find_packages(),
    include_package_data=True,

    author='Rishik Mourya',
    author_email='rishik@skylarklabs.ai',
    description='Clients library for SkylarkLabs.ai production',
    url=None,
    install_requires=install_requires,
    license='MIT',
)