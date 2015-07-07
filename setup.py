import setuptools

import versioneer

#import codecs
#with codecs.open('DESCRIPTION.rst', 'r', 'utf-8') as f:
#    long_description = f.read()

setuptools.setup(
    name="disc_solver",
    version=versioneer.get_version(),
    packages = ["disc_solver", "disc_solver.analyse", "disc_solver.solve"],
    install_requires = [
        "numpy",
        "matplotlib",
        "scikits.odes",
        "logbook",
        "arrow",
        "hdf5_wrapper",
    ],
    author = "James Tocknell",
    author_email = "aragilar@gmail.com",
    description = "Solver thing",
#    long_description = long_description,
    #license = "BSD",
    #keywords = "wheel",
    #url = "http://disc_solver.rtfd.org",
    #classifiers=[
    #    'Development Status :: 3 - Alpha',
    #    'Intended Audience :: Developers',
    #    "Topic :: System :: Shells",
    #    'License :: OSI Approved :: BSD License',
    #    'Programming Language :: Python :: 2',
    #    'Programming Language :: Python :: 2.6',
    #    'Programming Language :: Python :: 2.7',
    #    'Programming Language :: Python :: 3',
    #    'Programming Language :: Python :: 3.1',
    #    'Programming Language :: Python :: 3.2',
    #    'Programming Language :: Python :: 3.3',
    #    'Programming Language :: Python :: 3.4',
    #],
    entry_points = {
        'console_scripts': [
            "ds-soln = disc_solver:solution_main",
            "ds-analyse = disc_solver:analyse_main",
            "ds-quick = disc_solver:main",
        ],
    },
    cmdclass=versioneer.get_cmdclass(),
)
