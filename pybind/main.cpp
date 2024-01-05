#include <pybind11/pybind11.h>
#include <opencv2/opencv.hpp>
#include <string>

#include "thirdparty/ndarray_converter.h"
#include <blob_counter.h>

namespace py = pybind11;


PYBIND11_MODULE(cpp_blob_counter, m) {

  NDArrayConverter::init_numpy();

  py::class_<CppBlobCounter>(m, "PyCppBlobCounter")
  .def(py::init<std::string, bool, bool, bool,bool, int>())
  .def("read_image", &CppBlobCounter::read_image, "A function reading an image")
  .def("detect_blobs", &CppBlobCounter::detectBlob, "A function for deteting blobs")
  ;

}

