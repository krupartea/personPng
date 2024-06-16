// utils.h
#ifndef UTILS_H
#define UTILS_H

#include <string>

std::string OpenFileDialog();
std::string SaveFileDialog();
bool LoadTextureFromMat(const cv::Mat& image, ID3D11ShaderResourceView** out_srv, int* out_width, int* out_height, ID3D11Device* g_pd3dDevice);


#endif // UTILS_H