from setuptools import Extension, setup

setup(
    name="spiral_numeric_cpp",
    ext_modules=[
        Extension(
            "spiral_numeric_cpp",
            sources=["native/cpp/spiral_numeric_cpp_module.c"],
        )
    ],
)
