import setuptools

import versioneer

#import codecs
#with codecs.open('DESCRIPTION.rst', 'r', 'utf-8') as f:
#    long_description = f.read()

setuptools.setup(
    name = "disc_solver",
    version = versioneer.get_version(),
    packages = setuptools.find_packages('src'),
    package_dir = {'': 'src'},
    install_requires = [
        "numpy",
        "matplotlib>=2.0",
        "scikits.odes>=2.3.0dev0",
        "logbook",
        "arrow",
        "h5py>2.5",
        "h5preserve>=0.14",
        "stringtopy",
        "corner",
        "attrs",
        "emcee",
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
            "ds-soln = disc_solver.solve:main",
            "ds-resoln = disc_solver.solve.resolve:main",
            "ds-info = disc_solver.analyse.info:info_main",
            "ds-plot = disc_solver.analyse.plot:plot_main",
            "ds-derivs-plot = disc_solver.analyse.derivs_plot:derivs_main",
            "ds-params-plot = disc_solver.analyse.params_plot:params_main",
            "ds-taylor-plot = disc_solver.analyse.taylor_plot:taylor_main",
            "ds-combine-plot = disc_solver.analyse.combine_plot:combine_main",
            "ds-jacobian-plot = disc_solver.analyse.jacobian_plot:jacobian_main",
            "ds-scale-height = disc_solver.analyse.scale_height:scale_height_main",
            "ds-validate-input-plot = disc_solver.analyse.validate_input_plot:validate_input_plot_main",
            "ds-acc-plot = disc_solver.analyse.acc_plot:acc_main",
            "ds-conserve-plot = disc_solver.analyse.conserve_plot:conserve_main",
            "ds-filter-files = disc_solver.filter_files:main",
        ],
    },
    cmdclass=versioneer.get_cmdclass(),
)
