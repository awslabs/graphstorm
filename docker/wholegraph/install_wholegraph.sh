#!/bin/bash
git clone https://github.com/fmtlib/fmt.git  /opt/fmt
cd /opt/fmt
git checkout 9.1.0
mkdir build && cd build
cmake -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE ..
make
make install

git clone https://github.com/gabime/spdlog.git  /opt/spdlog
cd /opt/spdlog && mkdir build && cd build
cmake .. && make -j
cp libspdlog.a  /usr/lib/libspdlog.a
export PYTHON=/usr/bin/python

cd  /opt/rapids/
git clone https://github.com/rapidsai/wholegraph.git -b branch-23.08
cd /opt/rapids/wholegraph/
pip install scikit-build
export WHOLEGRAPH_CMAKE_CUDA_ARCHITECTURES="70-real;80-real;90"
# fix a bug in CMakeList.txt when build pylibwholegraph
old="import sysconfig; print(sysconfig.get_config_var('BINLIBDEST'))"
string="import sysconfig; print(\"%s/%s\" % (sysconfig.get_config_var(\"LIBDIR\"), sysconfig.get_config_var(\"INSTSONAME\")))"
sed -i "s|$old|$string|" /opt/rapids/wholegraph/python/pylibwholegraph/CMakeLists.txt
bash build.sh libwholegraph pylibwholegraph -v
