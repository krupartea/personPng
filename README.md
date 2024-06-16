# personPng
![personPng_demo](https://github.com/krupartea/personPng/assets/48570933/558d5fd8-44c1-40ce-9123-47e8cab26969)

This is a C++ program which removes background on images with people, allowing to save a background-free `.png` image.

The code is written using primarily PyTorch, ONNX, OpenCV, and ImGui frameworks.

## Noticable files
- `export_model.py` script creates an `.onnx` file. DeepLabV3 is used as a backend model. Its outputs are slightly altered in the script, to leave less headache with obtaining the alpha channel in C++.
- `main.cpp` contains the entrypoint, and the inference logic.

## Big thanks to
- [ImGui](https://github.com/ocornut/imgui) and [this](https://github.com/ocornut/imgui/wiki/Image-Loading-and-Displaying-Examples) ImGui tutorial on displaying images.
- [This](https://youtu.be/imjqRdsm2Qw?si=ljYNL69ycvWB6tOW) YouTube tutorial on using ONNX in C++.
