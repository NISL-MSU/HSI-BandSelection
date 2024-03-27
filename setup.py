import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='HSIBandSelection',
    version='0.1.1',
    author='Giorgio Morales - Montana State University',
    author_email='giorgiomoralesluna@gmail.com',
    description='Developing Low-Cost Multispectral Imagers using Inter-Band Redundancy Analysis and Greedy Spectral Selection in Hyperspectral Imaging',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/NISL-MSU/HSI-BandSelection',
    project_urls={"Bug Tracker": "https://github.com/NISL-MSU/HSI-BandSelection/issues"},
    license='MIT',
    packages=setuptools.find_packages('src', exclude=['test']),
    # packages=setuptools.find_namespace_packages(where="src", exclude=['test']),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=['matplotlib', 'numpy', 'opencv-python', 'statsmodels', 'tqdm', 'timeout_decorator',
                      'h5py', 'pyodbc', 'regex', 'torchsummary', 'python-dotenv', 'omegaconf', 'pandas'],
    package_data={'HSIBandSelection.Data': ['*.mat']}
)
