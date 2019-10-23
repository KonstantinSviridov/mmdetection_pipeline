from setuptools import setup
import setuptools
setup(name='mmdetection_pipeline',
      version='0.01',
      description='MMDetection based instance segmentation pipeline',
      url='https://github.com/musket-ml/instance_segmentation_pipeline/',
      author='Petrochenko Pavel',
      author_email='petrochenko.pavel.a@gmail.com',
      license='MIT',
      packages=setuptools.find_packages(),
      include_package_data=True,
      dependency_links=['https://github.com/aleju/imgaug'],
      install_requires=["numpy", "scipy","Pillow", "cython","pandas","matplotlib", "scikit-image","keras>=2.2.4","imageio",
"opencv-python",
"h5py",
"tqdm",
"segmentation_models", "lightgbm", "async_promises"],
      zip_safe=False)