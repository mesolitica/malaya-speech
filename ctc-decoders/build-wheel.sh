#!/bin/bash
# https://github.com/pypa/python-manylinux-demo/blob/master/travis/build-wheels.sh
set -e -u -x

cd /io
mkdir -p wheelhouse

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w /io/wheelhouse/
    fi
}

/opt/python/cp36-cp36m/bin/pip wheel /io/ --no-deps -w wheelhouse/
/opt/python/cp37-cp37m/bin/pip wheel /io/ --no-deps -w wheelhouse/
/opt/python/cp38-cp38/bin/pip wheel /io/ --no-deps -w wheelhouse/

for whl in wheelhouse/*.whl; do
    repair_wheel "$whl"
done
