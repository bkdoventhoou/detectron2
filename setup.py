#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import glob
import os
import shutil
from os import path
from setuptools import find_packages, setup
from typing import List
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 8], "Requires PyTorch >= 1.8"


def get_version():
    init_py_path = path.join(path.abspath(path.dirname(__file__)), "detectron2", "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip('"')
    return version


def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "detectron2", "layers", "csrc")

    main_source_dir = extensions_dir
    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"), recursive=True)

    from torch.utils.cpp_extension import ROCM_HOME

    is_rocm_pytorch = (
        True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False
    )
    is_cuda = CUDA_HOME is not None

    extension = CppExtension

    extra_compile_args = {"cxx": ["-O3", "-std=c++17"]}
    define_macros = []

    if (is_cuda and (torch_ver >= [1, 7])) or is_rocm_pytorch:
        extension = CUDAExtension
        sources += glob.glob(path.join(extensions_dir, "**", "*.cu"), recursive=True)

        if not is_rocm_pytorch:
            define_macros += [("WITH_CUDA", None)]
            extra_compile_args["nvcc"] = [
                "-O3",
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ]
        else:
            define_macros += [("WITH_HIP", None)]
            extra_compile_args["nvcc"] = []

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "detectron2._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


def get_model_zoo_configs() -> List[str]:
    """Return a list of configs to include in the package."""
    source_configs_dir = path.join(path.dirname(path.realpath(__file__)), "configs")
    destination = path.join(
        path.dirname(path.realpath(__file__)), "detectron2", "model_zoo", "configs"
    )
    if path.exists(destination):
        shutil.rmtree(destination)
    try:
        shutil.copytree(source_configs_dir, destination)
    except Exception:
        pass
    config_paths = glob.glob("configs/**/*.yaml", recursive=True) + glob.glob(
        "configs/**/*.py", recursive=True
    )
    return config_paths


setup(
    name="detectron2",
    version=get_version(),
    author="FAIR",
    url="https://github.com/facebookresearch/detectron2",
    description="Detectron2 is Facebook AI Research's next generation library"
    " that provides state-of-the-art detection and segmentation algorithms.",
    packages=find_packages(exclude=("configs", "tests", "*.tests", "*.tests.*", "tests.*")),
    package_data={"detectron2.model_zoo": get_model_zoo_configs()},
    python_requires=">=3.7",
    install_requires=[
        "Pillow>=7.1",
        "matplotlib",
        "pycocotools>=2.0.2",
        "termcolor>=1.1",
        "yacs>=0.1.8",
        "tabulate",
        "cloudpickle",
        "tqdm>4.29.0",
        "tensorboard",
        "fvcore>=0.1.5,<0.1.6",
        "iopath>=0.1.7,<0.1.10",
        "omegaconf>=2.1,<2.4",
        "hydra-core>=1.1",
        "black",
        "packaging",
    ],
    extras_require={
        "all": [
            "fairscale",
            "timm",
            "scipy>1.5.1",
            "shapely",
            "pygments>=2.2",
        ],
        "dev": [
            "flake8==3.8.1",
            "isort==4.3.21",
            "flake8-bugbear",
            "flake8-comprehensions",
        ],
    },
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
