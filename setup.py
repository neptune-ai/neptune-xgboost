import os

from neptune_xgboost._version import get_versions
from setuptools import setup


def main():
    with open('README.md') as readme_file:
        readme = readme_file.read()

    extras = {}

    all_deps = []
    for group_name in extras:
        all_deps += extras[group_name]
    extras['all'] = all_deps

    base_libs = ['neptune-client>=0.9.8', 'xgboost>=1.3.0', 'matplotlib', 'graphviz']

    version = None
    if os.path.exists('PKG-INFO'):
        with open('PKG-INFO', 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith('Version:'):
                version = line[8:].strip()
    else:
        version = get_versions()["version"]

    setup(
        name='neptune-xgboost',
        version=version,
        description='Neptune.ai XGBoost integration library',
        author='neptune.ai',
        support='contact@neptune.ai',
        author_email='contact@neptune.ai',
        url="https://github.com/neptune-ai/neptune-xgboost",
        project_urls={
            'Tracker': 'https://github.com/neptune-ai/neptune-xgboost/issues',
            'Source': 'https://github.com/neptune-ai/neptune-xgboost',
            'Documentation': 'https://docs.neptune.ai/integrations-and-supported-tools/model-training/xgboost',
        },
        long_description=readme,
        long_description_content_type="text/markdown",
        license='Apache License 2.0',
        install_requires=base_libs,
        extras_require=extras,
        packages=['neptune_xgboost', 'neptune_xgboost.impl'],
        zip_safe=False,
        classifiers=[
            # As from http://pypi.python.org/pypi?%3Aaction=list_classifiers
            'Development Status :: 4 - Beta',
            # 'Development Status :: 5 - Production/Stable',  # Switch to Stable when applicable
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Natural Language :: English',
            'Operating System :: MacOS',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Programming Language :: Python :: Implementation :: CPython',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
        keywords=['MLOps', 'ML Experiment Tracking', 'ML Model Registry', 'ML Model Store', 'ML Metadata Store'],
    )


if __name__ == "__main__":
    main()
