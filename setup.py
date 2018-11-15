import setuptools

import versioneer

import codecs
with codecs.open('DESCRIPTION.rst', 'r', 'utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name = "disc-solver",
    version = versioneer.get_version(),
    packages = setuptools.find_packages('src'),
    package_dir = {'': 'src'},
    install_requires = [
        "numpy",
        "matplotlib>=2.2",
        "scikits.odes>=2.3.0dev0",
        "logbook",
        "arrow",
        "h5py>2.5",
        "h5preserve>=0.14",
        "stringtopy",
        "corner",
        "attrs",
        "emcee",
        "spaceplot",
        "root-solver",
    ],
    author = "James Tocknell",
    author_email = "aragilar@gmail.com",
    description = "Solver for jet solutions in PPDs",
    long_description = long_description,
    license = "GPLv3+",
    url = "http://disc-solver.rtfd.org",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    entry_points = {
        'console_scripts': [
            "ds-soln = disc_solver.solve:main",
            "ds-resoln = disc_solver.solve.resolve:main",
            "ds-info = disc_solver.analyse.info:info_main",
            "ds-plot = disc_solver.analyse.plot:plot_main",
            "ds-compare-plot = disc_solver.analyse.compare_plot:plot_main",
            "ds-derivs-plot = disc_solver.analyse.derivs_plot:derivs_main",
            "ds-params-plot = disc_solver.analyse.params_plot:params_main",
            "ds-taylor-plot = disc_solver.analyse.taylor_plot:taylor_main",
            "ds-combine-plot = disc_solver.analyse.combine_plot:combine_main",
            "ds-validate-plot = disc_solver.analyse.validate_plot:validate_plot_main",
            "ds-hydro-check-plot = disc_solver.analyse.hydro_check_plot:hydro_check_plot_main",
            "ds-acc-plot = disc_solver.analyse.acc_plot:acc_main",
            "ds-diverge-plot = disc_solver.analyse.diverge_plot:diverge_main",
            "ds-conserve-plot = disc_solver.analyse.conserve_plot:conserve_main",
            "ds-j-e-plot = disc_solver.analyse.j_e_plot:j_e_plot_main",
            "ds-plot-taylor-space = disc_solver.solve.taylor_space:main",
            "ds-component-plot = disc_solver.analyse.component_plot:plot_main",
        ],
    },
    cmdclass=versioneer.get_cmdclass(),
)
