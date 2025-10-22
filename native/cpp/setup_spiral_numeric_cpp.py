from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


setup(
    name="spiral_numeric_cpp",
    ext_modules=[
        Pybind11Extension(
            "spiral_numeric_cpp",
            ["native/cpp/spiral_numeric_cpp_module.cpp"],
            cxx_std=17,
        )
    ],
    cmdclass={"build_ext": build_ext},
)
