#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "Helpers.cpp"  // preprocessing for inference
#include <cmath>
#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"
#include <d3d11.h>
#include <tchar.h>
#include "utils.h"  // file dialogues, texture loading
#include <stdio.h>


// d3d11 data
static ID3D11Device*            g_pd3dDevice = nullptr;
static ID3D11DeviceContext*     g_pd3dDeviceContext = nullptr;
static IDXGISwapChain*          g_pSwapChain = nullptr;
static bool                     g_SwapChainOccluded = false;
static UINT                     g_ResizeWidth = 0, g_ResizeHeight = 0;
static ID3D11RenderTargetView*  g_mainRenderTargetView = nullptr;

// Forward declarations of helper functions
bool CreateDeviceD3D(HWND hWnd);
void CleanupDeviceD3D();
void CreateRenderTarget();
void CleanupRenderTarget();
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);


int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
    PSTR lpCmdLine, int nCmdShow)
{

    // Create application window
    WNDCLASSEXW wc = { sizeof(wc), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr, L"ImGui Example", nullptr };
    ::RegisterClassExW(&wc);
    HWND hwnd = ::CreateWindowW(wc.lpszClassName, L"personPng", WS_OVERLAPPEDWINDOW, 100, 100, 500, 400, nullptr, nullptr, wc.hInstance, nullptr);

    // Initialize Direct3D
    if (!CreateDeviceD3D(hwnd))
    {
        CleanupDeviceD3D();
        ::UnregisterClassW(wc.lpszClassName, wc.hInstance);
        return 1;
    }

    // Show the window
    ::ShowWindow(hwnd, SW_SHOWDEFAULT);
    ::UpdateWindow(hwnd);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    // ImGui::StyleColorsDark();
    ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);


    // background color
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);


    // set up initial state...
    // ...of control flow
    bool done = false;  // imgui loop
    bool inference = false;  // inference should be called on the loaded image
    bool show = false;  // `my_texture` should be displayed
    bool inferenceDone = false;  // there was a successful inference 
    // ...of inference and display variable
    std::string imgPath = "Path";
    cv::Mat rgba;  // output image
    ID3D11ShaderResourceView* my_texture = NULL;  // data to display
    int my_image_height = 0;
    int my_image_width = 0;
    
    int displayHeight = 240;  // all displayed images will be resized to have this height

    // imgui loop
    while (!done)
    {
        // Poll and handle messages (inputs, window resize, etc.)
        // See the WndProc() function below for our to dispatch events to the Win32 backend.
        MSG msg;
        while (::PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE))
        {
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);
            if (msg.message == WM_QUIT)
                done = true;
        }
        if (done)
            break;

        // Handle window being minimized or screen locked
        if (g_SwapChainOccluded && g_pSwapChain->Present(0, DXGI_PRESENT_TEST) == DXGI_STATUS_OCCLUDED)
        {
            ::Sleep(10);
            continue;
        }
        g_SwapChainOccluded = false;

        // Handle window resize (we don't resize directly in the WM_SIZE handler)
        if (g_ResizeWidth != 0 && g_ResizeHeight != 0)
        {
            CleanupRenderTarget();
            g_pSwapChain->ResizeBuffers(0, g_ResizeWidth, g_ResizeHeight, DXGI_FORMAT_UNKNOWN, 0);
            g_ResizeWidth = g_ResizeHeight = 0;
            CreateRenderTarget();
        }

        // Start the Dear ImGui frame
        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();
        ImGui::SetNextWindowSize(ImVec2(480, 360), ImGuiCond_Always);
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);

        // imgui boilerplate ends here
        
        ImGui::Begin("personPng");

        // display a selected image, store its path
        // TODO: make the image stored (not path), to avoid loading it for inference again
        if (ImGui::Button("Open image")){
            imgPath = OpenFileDialog();
            cv::Mat imageForDisplay;
            imageForDisplay = cv::imread(imgPath);
            cv::cvtColor(imageForDisplay, imageForDisplay, cv::COLOR_BGR2RGBA);
            // compute resized aspect-ratio-preserving width
            // TODO: wrap into a function, since also used to display the inference result
            int displayWidth = static_cast<int>((float)imageForDisplay.cols * ((float)displayHeight / (float)imageForDisplay.rows));
            cv::resize(imageForDisplay, imageForDisplay, cv::Size(displayWidth, displayHeight));
            // update `my_texture` with the `imageForDisplay`'s data
            bool ret = LoadTextureFromMat(imageForDisplay, &my_texture, &my_image_width, &my_image_height, g_pd3dDevice);
            show = true;
        }

        // start inference button
        ImGui::SameLine();  // keep buttons on same line
        if (show){
            if (ImGui::Button("Process")){
                inference = true;
            }
        }   

        // save button
        ImGui::SameLine();
        if (inferenceDone){
            if (ImGui::Button("Save")){
                std::string savePath = SaveFileDialog();
                cv::imwrite(savePath, rgba);
            }
        }

        // show image
        if (show){
            ImGui::NewLine();  // indent from buttons row
            ImGui::Image((void*)my_texture, ImVec2(my_image_width, my_image_height));
        }


        if (inference){
            Ort::Env env;
            Ort::RunOptions runOptions;
            Ort::Session session(nullptr);

            constexpr int64_t modelInputNumChannels = 3;
            constexpr int64_t modelInputWidth = 520;
            constexpr int64_t modelInputHeight = 520;
            constexpr int64_t modelInputNumElements = modelInputNumChannels * modelInputHeight * modelInputWidth;

            // TODO: for some reason only absolute path works. Fix to enable relative paths
            auto modelPath = L"model4.onnx";  // rn it's a placeholder-fake path


            // load input image without changes and keep to concat with predicted alpha
            cv::Mat imageOrig = cv::imread(imgPath);
            if (imageOrig.empty()) {
                std::cout << "No image found.";
            }

            // load image and preprocess it wrt DeepLab transforms (Imagenet's standardization)
            const std::vector<float> imageVec = loadImage(imgPath, modelInputHeight, modelInputWidth);
            if (imageVec.empty()) {
                std::cout << "Failed to load image: " << imgPath << std::endl;
                return 1;
            }

            // NOTE: at this point there are two redundant image from-disk readings
            // TODO: initally loaded (for display) image must be kept and used for inference

            // Use CPU
            session = Ort::Session(env, modelPath, Ort::SessionOptions{ nullptr });
            
            // prepare input and output tensors
            // define shape
            const std::array<int64_t, 4> inputShape = { 1, modelInputNumChannels, modelInputHeight, modelInputWidth };
            const std::array<int64_t, 3> outputShape = { 1, modelInputHeight, modelInputWidth };
            // // define array
            size_t inputSize = 4 * modelInputNumChannels * modelInputHeight * modelInputWidth;
            size_t resultsSize = 4 * modelInputHeight * modelInputWidth;
            float* input = (float*)malloc(inputSize);
            float* results = (float*)malloc(resultsSize);
            // define Tensor
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
            auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, input, inputSize, inputShape.data(), inputShape.size());
            auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, results, resultsSize, outputShape.data(), outputShape.size());
            // copy input image data to input array
            std::copy(imageVec.begin(), imageVec.end(), input);
            // define names
            Ort::AllocatorWithDefaultOptions ort_alloc;
            Ort::AllocatedStringPtr inputName = session.GetInputNameAllocated(0, ort_alloc);
            Ort::AllocatedStringPtr outputName = session.GetOutputNameAllocated(0, ort_alloc);
            const std::array<const char*, 1> inputNames = { inputName.get()};
            const std::array<const char*, 1> outputNames = { outputName.get()};
            inputName.release();
            outputName.release();


            // run inference
            try {
                session.Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
            }
            catch (Ort::Exception& e) {
                std::cout << e.what() << std::endl;
            }


            // combine predicted alpha with the original image to obtain rgba
            cv::Mat alpha(modelInputHeight, modelInputWidth, CV_32FC1, results);
            cv::resize(alpha, alpha, cv::Size(imageOrig.cols, imageOrig.rows), 0);
            alpha.convertTo(alpha, CV_8U, 255);
            // original image doesn't have an alpha-channel, so create a dummy one
            cv::cvtColor(imageOrig, rgba , cv::COLOR_RGB2RGBA);
            std::vector<cv::Mat>channels(4);
            cv::split(rgba, channels);
            channels[3] = alpha;
            cv::merge(channels, rgba);
            cv::cvtColor(rgba, rgba , cv::COLOR_BGRA2RGBA);

            // display the result
            cv::Mat imageForDisplay;
            int displayWidth = static_cast<int>((float)rgba.cols * ((float)displayHeight / (float)rgba.rows));
            cv::resize(rgba, imageForDisplay, cv::Size(displayWidth, displayHeight));
            bool ret = LoadTextureFromMat((const cv::Mat&)imageForDisplay, &my_texture, &my_image_width, &my_image_height, g_pd3dDevice);
            cv::cvtColor(rgba, rgba, cv::COLOR_BGRA2RGBA);

            // update control flow
            inference = false;
            inferenceDone = true;
        }


        ImGui::End();


        // Rendering
        ImGui::Render();
        const float clear_color_with_alpha[4] = { clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w };
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, nullptr);
        g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, clear_color_with_alpha);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

        // Present
        HRESULT hr = g_pSwapChain->Present(1, 0);   // Present with vsync
        //HRESULT hr = g_pSwapChain->Present(0, 0); // Present without vsync
        g_SwapChainOccluded = (hr == DXGI_STATUS_OCCLUDED);
    }

    // Cleanup
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    CleanupDeviceD3D();
    ::DestroyWindow(hwnd);
    ::UnregisterClassW(wc.lpszClassName, wc.hInstance);

    return 0;
}


// Helper functions

bool CreateDeviceD3D(HWND hWnd)
{
    // Setup swap chain
    DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory(&sd, sizeof(sd));
    sd.BufferCount = 2;
    sd.BufferDesc.Width = 0;
    sd.BufferDesc.Height = 0;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    UINT createDeviceFlags = 0;
    //createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
    D3D_FEATURE_LEVEL featureLevel;
    const D3D_FEATURE_LEVEL featureLevelArray[2] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0, };
    HRESULT res = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, createDeviceFlags, featureLevelArray, 2, D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext);
    if (res == DXGI_ERROR_UNSUPPORTED) // Try high-performance WARP software driver if hardware is not available.
        res = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, createDeviceFlags, featureLevelArray, 2, D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext);
    if (res != S_OK)
        return false;

    CreateRenderTarget();
    return true;
}

void CleanupDeviceD3D()
{
    CleanupRenderTarget();
    if (g_pSwapChain) { g_pSwapChain->Release(); g_pSwapChain = nullptr; }
    if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = nullptr; }
    if (g_pd3dDevice) { g_pd3dDevice->Release(); g_pd3dDevice = nullptr; }
}

void CreateRenderTarget()
{
    ID3D11Texture2D* pBackBuffer;
    g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
    g_pd3dDevice->CreateRenderTargetView(pBackBuffer, nullptr, &g_mainRenderTargetView);
    pBackBuffer->Release();
}

void CleanupRenderTarget()
{
    if (g_mainRenderTargetView) { g_mainRenderTargetView->Release(); g_mainRenderTargetView = nullptr; }
}

// Forward declare message handler from imgui_impl_win32.cpp
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Win32 message handler
// You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
// - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
// - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
// Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch (msg)
    {
    case WM_SIZE:
        if (wParam == SIZE_MINIMIZED)
            return 0;
        g_ResizeWidth = (UINT)LOWORD(lParam); // Queue resize
        g_ResizeHeight = (UINT)HIWORD(lParam);
        return 0;
    case WM_SYSCOMMAND:
        if ((wParam & 0xfff0) == SC_KEYMENU) // Disable ALT application menu
            return 0;
        break;
    case WM_DESTROY:
        ::PostQuitMessage(0);
        return 0;
    }
    return ::DefWindowProcW(hWnd, msg, wParam, lParam);
}
