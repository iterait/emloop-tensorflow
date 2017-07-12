from pip.req import parse_requirements
from setuptools import setup

setup(name='cxflow-tf',
      version='0.1',
      description='TensorFlow plugin for cxflow.',
      long_description='Plugin that enables cxflow to work with TensorFlow.',
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
      url='https://github.com/Cognexa/cxflow-tensorboard',
      author='Petr Belohlavek',
      author_email='me@petrbel.cz',
      license='MIT',
      packages=[
          'cxflow_tf',
          'cxflow_tf.hooks',
          'cxflow_tf.third_party',
          'cxflow_tf.third_party.tensorflow'
      ],
      include_package_data=True,
      zip_safe=False,
      test_suite='cxflow_tf.tests',
      install_requires=[str(ir.req) for ir in parse_requirements('requirements.txt', session='hack')],
     )
