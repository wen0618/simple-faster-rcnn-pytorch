from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
#构建动态链接库的代码，Cython标准写法，详见Cython，这里就是将我们写的_nms_gpu_post.pyx变成pyd
#或.so文件的过程 所以我们要先运行这个类，运行方法作者也介绍的很清楚了，最终生成pyd or .so链接库 .c是中间文件



#ext_modules = [Extension("_nms_gpu_post", ["_nms_gpu_post.pyx"])]
ext_modules = [Extension("_nms_gpu_post", ["_nms_gpu_post.pyx"],
 include_dirs=[numpy.get_include()])]
setup(
    name="nms pyx",
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
