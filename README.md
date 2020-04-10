# Neural Art in PyTorch
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/open?id=1_1bqYr-A5yK7dae-Hqvbho0e6CnUXOVU)

This repository cotains implementation of Neural Art using PyTorch. Neural Art is also famous with the name of Neural Style Transfer. In this repository , I have used VGG16 pretrained model for extracting features. You can read the paper [here](https://www.cvfoundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)


## Instructions for using C++ scipts
  For using this project you need to follow some steps:
  
   1. First clone this repository in your system and then navigate to repository folder
    
      ``` 
      git clone https://github.com/adityak2920/NeuralArt-in-pytorch.git
      cd NeuralArt-in-pytorch/cpp
      ``` 
   2. Create a folder build 
    
      ```
      mkdir build && cd build
      ```
   3. Before procedding further we need to download latest distribution of libtorch to build our application which you can download from [here](https://pytorch.org/). After downloading, unzip the folder and your libtorch directory should looks like this:
   
      ```
      libtorch/
              bin/
              include/
              lib/
              share/
              build-hash
              build-version
      ```
   4. Now run these commands to build the application
      ```
      cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
      cmake --build . --config Release
      ```
   5. To run our app, we will need vgg19 to torchscript form:
      ```
          import torchvision
          import torch
          model = torchvision.models.vgg19(pretrained=True)
          x = torch.rand(1, 3, 448, 448)
          traced_cell = torch.jit.trace(model, x)
          traced_cell.save('model.pt')
      ```
      Now use the path of saved model in neural.cpp .
          
      
   6. Now you can run your app with(you can specify name of app in cmake file)
      ```
        ./your_app
      ```
      
   7. From next time you can recompile and run your app using
      ```
      make
      ./your_app
      ```


Here, are some of the results after training for some time using Tesla K80 GPU on Google Collab:

<img src="https://user-images.githubusercontent.com/35501699/54220463-7c17a700-4517-11e9-8256-6c0c2f396ff9.jpg" width="400" height="400">     <img src="https://user-images.githubusercontent.com/35501699/54220469-7de16a80-4517-11e9-9912-3e5384929ec8.jpg" width="400" height="400"> 
          <img src="https://user-images.githubusercontent.com/35501699/54220476-80dc5b00-4517-11e9-8443-f6ba00eb947e.jpg" width="450" height="450"> 



Here are some other images generated using same content image but with different style images and the results are pretty good. 

![neural2](https://user-images.githubusercontent.com/35501699/47318653-3e9aa600-d66a-11e8-80df-5ad4db36a75c.jpg)
![neural3](https://user-images.githubusercontent.com/35501699/47318655-3e9aa600-d66a-11e8-92dc-9af6e87fb418.jpg)
![neural4](https://user-images.githubusercontent.com/35501699/47318657-3f333c80-d66a-11e8-92bc-0432eb4bed2e.jpg)







