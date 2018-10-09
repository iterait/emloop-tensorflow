from setuptools import setup

setup(name='emloop-tensorflow',
      version='0.1.0',
      description='TensorFlow extension for emloop.',
      long_description='Plugin that enables emloop to work with TensorFlow.',
      classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: Unix',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
      ],
      keywords='tensorflow wrapper',
      url='https://github.com/iterait/emloop-tensorflow',
      author='Iterait a.s.',
      author_email='hello@iterait.com',
      license='MIT',
      packages=[
          'emloop_tensorflow',
          'emloop_tensorflow.hooks',
          'emloop_tensorflow.models',
          'emloop_tensorflow.ops',
          'emloop_tensorflow.utils',
          'emloop_tensorflow.metrics',
          'emloop_tensorflow.third_party',
          'emloop_tensorflow.third_party.tensorflow'
      ],
      include_package_data=True,
      zip_safe=False,
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      install_requires=[line for line in open('requirements.txt', 'r').readlines() if not line.startswith('#')],
     )
