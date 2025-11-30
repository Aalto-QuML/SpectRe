from setuptools import setup, Extension
from torch.utils import cpp_extension

torch_library_paths = cpp_extension.library_paths(cuda=False)

setup(name='spectre_lobpcg',
      ext_modules=[cpp_extension.CppExtension('spectre_lobpcg', ['ph/spectre_lobpcg_cpu.cpp'], extra_link_args=[
                    '-Wl,-rpath,' + library_path
                    for library_path in torch_library_paths])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})