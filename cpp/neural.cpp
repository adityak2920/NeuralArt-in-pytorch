// including all the header files needed for running this script
#include <torch/torch.h>
#include <torch/script.h> 
#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>



using namespace cv;
using namespace std;

int main() {
    // Setting precision to 4 decimal places.
    std::cout << std::fixed << std::setprecision(4);

    // Loading the pretrained vgg model .
    torch::jit::script::Module model = torch::jit::load("path to model");

    // Initialising Normalization and De-Normalization transform for preprocessing
    torch::data::transforms::Normalize<> normalize_transform({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225});
    torch::data::transforms::Normalize<> denormalize_transform({-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225}, {1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225});

    // Creating a variabe CUDA for running on GPU
    torch::Device device(torch::kCUDA);


    // loading content image and doing some pre-processing   
    Mat image_bgr, image_bgr1, content_image, style_image;
    image_bgr = imread("path to content image");
    // As OpenCV reads image in BGR so converting to RGB, normaizing and then resizing the image
    cvtColor(image_bgr, image_bgr, COLOR_BGR2RGB);
    image_bgr.convertTo(image_bgr, CV_32FC3, 1.0f / 255.0f);
    resize(image_bgr, content_image, {448, 448}, INTER_NEAREST);


    // loading content image and doing some pre-processing   
    image_bgr1 = imread("path to style image");
    // As OpenCV reads image in BGR so converting to RGB, normaizing and then resizing the image
    cvtColor(image_bgr1, image_bgr1, COLOR_BGR2RGB);
    image_bgr1.convertTo(image_bgr1, CV_32FC3, 1.0f / 255.0f);
    resize(image_bgr1, style_image, {448, 448}, INTER_NEAREST);



    // Preparing Data for Content Image
    auto content = torch::from_blob(content_image.data, {content_image.rows, content_image.cols, 3});
    content = content.permute({2, 0, 1});
    torch::Tensor content = normalize_transform(content).unsqueeze_(0);
    std::vector<torch::jit::IValue> content_tensor;
    content_tensor.push_back(content.to(device));

    // Preparing Data for Style Image
    auto style = torch::from_blob(style_image.data, {style_image.rows, style_image.cols, 3});
    style = style.permute({2, 0, 1});
    torch::Tensor style = normalize_transform(style).unsqueeze_(0);
    std::vector<torch::jit::IValue> style_tensor;
    style_tensor.push_back(style.to(device));


    auto target_tensor = content_tensor.clone();

    // Shifting the model from training to evaluation model, then shifting the model to CUDA(GPU).
    model.eval();
    model.to(device);

    // Initialising Optimizer
    torch::optim::Adam optimizer(std::vector<torch::Tensor>{target_tensor.requires_grad_(true)}, torch::optim::AdamOptions(3e-3).beta1(0.5).beta2(0.999));


    cout<<"Starting Training"<<"\n";
    for (size_t step = 0; step != 2000; ++step) {
        // Forward pass and extract feature tensors from some Conv2d layers
        auto target_features = model.forward(target_tensor);
        auto content_features = model.forward(content_tensor);
        auto style_features = model.forward(style_tensor);

        auto style_loss = torch::zeros({1}, torch::TensorOptions(device));
        auto content_loss = torch::zeros({1}, torch::TensorOptions(device));

        for (size_t f_id = 0; f_id != target_features.size(); ++f_id) {
            // Compute content loss between target and content feature images
            content_loss += torch::nn::functional::mse_loss(target_features[f_id], content_features[f_id]);

            auto c = target_features[f_id].size(1);
            auto h = target_features[f_id].size(2);
            auto w = target_features[f_id].size(3);

            // Reshape convolutional feature maps
            auto target_feature = target_features[f_id].view({c, h * w});
            auto style_feature = style_features[f_id].view({c, h * w});

            // Compute gram matrices
            target_feature = torch::mm(target_feature, target_feature.t());
            style_feature = torch::mm(style_feature, style_feature.t());

            // Compute style loss
            style_loss += torch::nn::functional::mse_loss(target_feature, style_feature) / (c * h * w);
        }

        // Compute total loss
        auto total_loss = content_loss + 100 * style_loss;

        // Backward pass and optimize
        optimizer.zero_grad();
        total_loss.backward();
        optimizer.step();

        if ((step + 1) % 10 == 0) {
            // Print losses
            std::cout << "Step [" << step + 1 << "/" << num_total_steps
                << "], Content Loss: " << content_loss.item<double>()
                << ", Style Loss: " << style_loss.item<double>() << "\n";
        }

        if ((step + 1) % 500 == 0) {
            // Save the generated image
            auto image = denormalize_transform(target.to(torch::kCPU).clone().squeeze(0)).clamp_(0, 1);
            save_image(image, "output/output-" + std::to_string(step + 1) + ".png", 1, 0);
        }
    }

    std::cout << "Training finished!\n";
}

