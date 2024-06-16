#include <windows.h>
#include <commdlg.h>
#include <string>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <d3d11.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>


std::string WideStringToString(const std::wstring& wideString) {
    // Get the length of the resulting string in bytes
    int bufferSize = WideCharToMultiByte(CP_UTF8, 0, wideString.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (bufferSize == 0) {
        // Handle error
        return "";
    }

    // Allocate buffer for the converted string
    std::string result(bufferSize, '\0');

    // Convert the wide string to a multibyte string
    WideCharToMultiByte(CP_UTF8, 0, wideString.c_str(), -1, &result[0], bufferSize, nullptr, nullptr);

    // Remove the null terminator from the end of the string
    result.resize(bufferSize - 1);

    return result;
}


std::string OpenFileDialog() {
    OPENFILENAMEW ofn;      // common dialog box structure
    wchar_t szFile[260];    // buffer for file name

    // Initialize OPENFILENAME
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = szFile;
    ofn.lpstrFile[0] = '\0'; // Initialize the file name buffer
    ofn.nMaxFile = sizeof(szFile) / sizeof(szFile[0]);
    ofn.lpstrFilter = L"All\0*.*\0Text\0*.TXT\0";
    ofn.nFilterIndex = 1;
    ofn.lpstrFileTitle = NULL;
    ofn.nMaxFileTitle = 0;
    ofn.lpstrInitialDir = NULL;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

    // Display the Open dialog box
    if (GetOpenFileNameW(&ofn) == TRUE) {
        return WideStringToString(ofn.lpstrFile);
    } else {
        return ""; // Return an empty string if no file is selected or an error occurs
    }
}


std::string SaveFileDialog() {
    OPENFILENAMEW ofn;      // common dialog box structure
    wchar_t szFile[260];    // buffer for file name

    // Initialize OPENFILENAME
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = szFile;
    ofn.lpstrFile[0] = '\0'; // Initialize the file name buffer
    ofn.nMaxFile = sizeof(szFile) / sizeof(szFile[0]);
    ofn.lpstrFilter = L"All\0*.*\0Text\0*.TXT\0";
    ofn.nFilterIndex = 1;
    ofn.lpstrFileTitle = NULL;
    ofn.nMaxFileTitle = 0;
    ofn.lpstrInitialDir = NULL;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT;

    // Display the Save dialog box
    if (GetSaveFileNameW(&ofn) == TRUE) {
        return WideStringToString(ofn.lpstrFile);
    } else {
        return ""; // Return an empty string if no file is selected or an error occurs
    }
}


bool LoadTextureFromMat(const cv::Mat& image, ID3D11ShaderResourceView** out_srv, int* out_width, int* out_height, ID3D11Device* g_pd3dDevice)
{
    if (image.empty() || image.channels() != 4)
        return false;


    int image_width = image.cols;
    int image_height = image.rows;
    *out_width = image_width;
    *out_height = image_height;

    // Create texture description
    D3D11_TEXTURE2D_DESC desc;
    ZeroMemory(&desc, sizeof(desc));
    desc.Width = image_width;
    desc.Height = image_height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = 0;

    // Set up the subresource data
    D3D11_SUBRESOURCE_DATA subResource;
    subResource.pSysMem = image.data;
    subResource.SysMemPitch = desc.Width * 4; // 4 bytes per pixel for RGBA
    subResource.SysMemSlicePitch = 0;

    // Create the texture
    ID3D11Texture2D* pTexture = nullptr;
    HRESULT hr = g_pd3dDevice->CreateTexture2D(&desc, &subResource, &pTexture);
    if (FAILED(hr))
        return false;

    // Create the texture view
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
    ZeroMemory(&srvDesc, sizeof(srvDesc));
    srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = desc.MipLevels;
    srvDesc.Texture2D.MostDetailedMip = 0;

    hr = g_pd3dDevice->CreateShaderResourceView(pTexture, &srvDesc, out_srv);
    pTexture->Release(); // Release the texture reference, as it is no longer needed after creating the view

    return SUCCEEDED(hr);
}
