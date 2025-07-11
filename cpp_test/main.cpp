#include <torch/extension.h>
#include <vector>

torch::Tensor add_tensors(torch::Tensor a, torch::Tensor b) {
  return a + b;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add", &add_tensors, "A simple function that adds two tensors");
}