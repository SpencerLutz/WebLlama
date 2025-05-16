# CMSC828 Final Project: WebLlama
### By Akshat Parikh and Spencer Lutz
## Directory Structure
This is how the directory structure of the project should look like after the downloaded tarball is extracted.
```
WebLlama
    |- WebGPT/
    |- WebLlama/
    |- cmsc828g_final_report.pdf
    |- README.md
```

1. **WebGPT/**: This subfolder contains all our modifications done to the [WebGPT](https://github.com/0hq/WebGPT) project to profile its compute shaders and gather results.
2. **WebLlama**: This subfolder contains all our code for the WebLlama project as a Typescript application. The code supports WebGPU profiling and basic inference with the model.
3. **cmsc828g_final_report.pdf**: This file is our final report for the project
4. **README.md**: Current file

## How to Run on Zaratan
For this project, we performed all our experiments and testing locally without Zaratan as this project does not require many GPUs and inference is done through the browser.

All the code can be ran locally as long as there is a basic graphics card on the local computer. For your graphics card, you will probably need the graphics driver e.g Metal, Direct3d installed. You will also need "node.js", "npm", and "python3" installed. Since node and npm were not on Zaratan, we also tested locally.

The following instruction can be used to run and reproduce the project locally or on Zaratan with "sudo" access to install the appropriate packages.
### WebGPT Profiling
1. Install Chrome Browser and Python3
    ```
    sudo apt-get install python3 chromium-browser
2. Go in Chrome to the following url and ensure the "Use graphics acceleration when avalable option" is enabled.
    ```
    chrome://settings/?search=acceleration
3. ```
    cd WebGPT/
4.  Start HTTP server
    ```
    python3 -m http.server 8000
5. Visit http://localhost:8000 and click the "Load GPT2 117M Model" button.

6. Open up the developer console (Ctrl+Shift+I) and go to "Console".

7. Click generate text and after sometime you should see profiling results per shader in the console.

### WebLlama Inference/Profiling
1. Install all the required packages
    ```
    sudo apt-get install python3 chromium-browser node.js npm
2. Go in Chrome to the following url and ensure the "Use graphics acceleration when avalable option" is enabled.
    ```
    chrome://settings/?search=acceleration
3. ```
    cd WebLlama/
4.  Install node packages
    ```
    npm i
5.  Start HTTP server
    ```
    npm run dev
5. Visit http://localhost:3000 and click the "Load Model" button.

6. Open up the developer console (Ctrl+Shift+I) and go to "Console".

7. Enter a prompt and click send. After sometime you should see profiling results per shader in the console as well as the output tokens from the model.
