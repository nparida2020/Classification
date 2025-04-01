<img width="693" alt="Image" src="https://github.com/user-attachments/assets/c927f9ed-897d-46b8-8eea-9b09d1192f52" />
<img width="689" alt="Image" src="https://github.com/user-attachments/assets/7e8ac8e2-72df-4c3f-aa80-4ab7f4775430" />
<img width="561" alt="Image" src="https://github.com/user-attachments/assets/7beab16b-f3a7-4005-b937-040bd6bb401f" />

CNN Layers and Parameters:

<img width="366" alt="Image" src="https://github.com/user-attachments/assets/208d7c2e-9f75-4d91-9597-58f69f1bff7c" />

EXecution Environment:

The Project is executed using tensorflow python packages in google colab build environment. 
Due to current version of the tensorflow requirements, the project canot be executed in lower CPU version. Tensorflow 2.19.6 need hisher CPU version.
The XLA_GPU and XLA_CPU version is comatible for the current tensorflow python packages.

Datasets: 3 different classes (benign,malignant,normal) -images, stored in data folders

1) Create a data folder in google drive under MyDrive folder : \content\drive\MyDrive\data
2) Copy all three folders under data folder to \content\drive\MyDrive\data in google drive
3) Copy  classification.ipnyb file to google drive content folder
4) Create a folder save_model under the \content\drive\MyDrive in google drive to save the model after training.

Model Deplopyment And Testing:

The model deployed in google vwertext AI platform and tested using test data.
The model deployed using project id : project-bc001
The vertex ai link given below:
https://console.cloud.google.com/vertex-ai/models/locations/us-central1/models/1137261160492433408/versions/1/deploy?hl=en&invt=AbtQYw&project=project-bc001

Model Test Result: 

The model tested using test images in vertext AI and  model acuracy (approximately) up to 65 to 75 % observed.

<img width="959" alt="Image" src="https://github.com/user-attachments/assets/e88a8d13-b434-42fa-a4fe-b19c77d2ea47" />

<img width="953" alt="Image" src="https://github.com/user-attachments/assets/76964cec-5097-41b4-ac81-6f0f51ee336e" />

<img width="947" alt="Image" src="https://github.com/user-attachments/assets/44a6d3c1-0e85-4deb-8f9b-f5884afca667" />

Code uploaded in Github and link provided below:

https://github.com/nparida2020/Classification/blob/main/README.md


PPT Link:

https://1drv.ms/p/c/6a4267724fe517d0/EbxwQQhcVY5Lvq6Nu7biYTwBxz40kpbGEKee25n0KIeLTg?e=6uANqe

Conclusion:

This study examines Deep Learning CNN model for breast cancer classification using tesorflow library after preprocessing dataset. Breast cancer is a prevalent disease affecting women worldwide, with machine-learning approaches potentially impacting early detection and prognosis. The disease is classified into two subtypes: invasive ductal carcinoma (IDC) and ductal carcinoma in situ (DCIS). Early detection is crucial for successful treatment, and appropriate screening technologies are essential. Advancements in artificial intelligence have made mammography more accurate, and deep learning models are being developed to recognize breast cancer in computerized mammograms. Convolutional neural networks and AI are emerging in healthcare to improve image processing and reduce human eye recognition. 
