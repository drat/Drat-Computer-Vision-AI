# DRAT COMPUTER VISION
Perpustakaan Computer Vision yang ringan untuk mengintegrasikan model pembelajaran mendalam dengan perangkat kamera.

# DAFTAR ISI
- [Pendahuluan] (#pendahuluan)
- [Pemberitahuan] (#pemberitahuan)
- [Persyaratan] (#persyaratan)
- [Petunjuk] (#instruksi)
- [Benchmarks] (#benchmarks)
- [To-Dos] (#to-dos)
- [Kutipan] (#kutipan)

# Pendahuluan
Aplikasi yang menggunakan model pembelajaran mesin untuk tugas-tugas visi telah berkembang pesat dalam beberapa tahun terakhir, dan dengan demikian kebutuhan untuk alat-alat yang mengintegrasikan antara ilmu data dan pipa rekayasa. ForesAI bertujuan untuk menjembatani kesenjangan antara keduanya dengan menyediakan perpustakaan dengan API sederhana bagi Anda untuk menerapkan model pembelajaran mesin Anda yang dibangun di pustaka populer langsung ke perangkat kamera Anda di berbagai platform perangkat keras. Dengan penekanan khusus pada kasus penggunaan robotik, ForesAI bertujuan untuk meminimalkan penggunaan sumber daya sehingga Anda dapat menjalankan model Anda pada sebanyak mungkin konfigurasi perangkat keras yang berbeda dan memberi Anda alat untuk menyiarkan keluaran Anda ke seluruh sistem AI Anda.

Ini adalah pekerjaan awal yang sedang berlangsung. Proyek ini berasal dari penelitian saya sendiri pada CNN yang efisien jadi saya menambahkan fitur / debugging sesuai kebutuhan. Silakan periksa [Untuk-Dos] (# to-dos) untuk beberapa tugas yang akan datang. Namun, saya mencari umpan balik karena saya ingin perpustakaan untuk mendukung kasus penggunaan lain juga. Jangan ragu untuk membuka masalah atau membuat permintaan tarik sesuai keinginan Anda - Saya mencari kontributor tambahan saat saya terus membangun di perpustakaan.

# Notice
DratAI mendukung tugas-tugas terkait visi seperti deteksi objek, segmentasi sematik, dan segmentasi contoh berdasarkan model yang relevan. API ini mengasumsikan Anda telah menyiapkan model yang telah dilatih sebelumnya. Untuk model TensorFlow saya, semua pelatihan / evaluasi dilakukan melalui [TensorFlow Object Detection API] (https://github.com/tensorflow/models/tree/master/research/object_detection). Saya akan memberikan skrip yang saya gunakan untuk pelatihan saya sendiri di bawah repo yang berbeda di masa depan, tetapi YMMV sebanyak itu tergantung pada konfigurasi Anda sendiri.

Awalnya perpustakaan dimaksudkan untuk mendukung TensorFlow saja, tetapi karena Anda dapat melihat ruang lingkupnya telah meningkat secara drastis seperti yang dituntut oleh penelitian saya sendiri. Saya sedang dalam proses membangun standar, perpustakaan-agnostic inferface untuk membuat membangun alur kesimpulan baru jauh lebih mudah. Dengan demikian, semua fungsi run_detection dalam file ops akan terdepresiasi di masa depan. Jangan ragu untuk melihat modul **model_loader** di bawah **inference** untuk mengetahui bagaimana hal itu dilakukan.

Mendukung tugas-tugas berikut :

## Deteksi Obyek / Segmentasi Instance
Model TensorFlow yang dilatih menggunakan [TensorFlow Object Detection API] (https://github.com/tensorflow/models/tree/master/research/object_detection). Lihat [Kebun Binatang Model] (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) Untuk daftar lengkapnya. Baik SSD-Mobilenet dan Mask-RCNN telah diuji.


## Deteksi Obyek
[Movidius NCS](https://github.com/movidius/ncsdk/) compiled models for object detection. See the [NC App Zoo](https://github.com/movidius/ncappzoo) for details. The Caffe-compiled version of SSD-Mobilenet was tested.

## Semantic Segmentation
All [PyTorch](http://pytorch.org/) models. Right now visualization only works for the cityscape dataset. I've included [ERFNet](https://github.com/Eromera/erfnet_pytorch), [Resnet18 8s](https://github.com/warmspringwinds/pytorch-segmentation-detection), and [Resnet34 8s](https://github.com/warmspringwinds/pytorch-segmentation-detection) for use.

Saya ingin mendukung perpustakaan tambahan juga jadi beri tahu saya jika Anda ingin membantu di endeaver ini!

# Requirements
Must haves:
- Python 3.5 or above (any other version will require minor fixes)
    - Following packages:
        - numpy
        - psutil
- OpenCV3 (your own build or pip)

For TensorFlow:
- Pillow (PIL) (For instance segmentation)
- [TensorFlow](https://www.tensorflow.org/)

For Movidius:
- [Movidius SDK](https://movidius.github.io/ncsdk/)

For PyTorch (Currently, only GPU mode w/CUDA is supported at this time):
- Pillow (PIL)
- [PyTorch](http://pytorch.org/)

For Darknet (YOLO object detectors):
- Build the library from the author's repo [here](https://github.com/pjreddie/darknet) and put the binary under the "darknet" folder. You'll also want to change the file paths in the cfg folder to match your own

# Instructions
If you don't have a model in mind, feel free to use this [slightly modified SSD-mobilenetv1 model](https://drive.google.com/drive/folders/1Cwy89QCs3R2dFRxZ85TZJZFBFMtTsl0D?usp=sharing) here to test out the object detection function. You'll need both folders extracted within the "ForesAI" folder.

There are two main ways to access the module. If you want to run ForesAI as a standalone module:

```
python main.py --config_path <CONFIG_PATH> 
```
Di mana CONFIG_PATH adalah file json dengan konfigurasi yang ditunjukkan dalam folder demo_configs. Jika Anda ingin menguji ini di laptop Anda ** webcam_benchmark.json ** akan menjadi pilihan pertama yang baik. Menambahkan tanda "--benchmark" akan menampilkan grafik yang mengukur penggunaan cpu / ram dari waktu ke waktu. Satu hal yang perlu diperhatikan adalah bahwa ** device_path ** dalam konfigurasi tidak harus menjadi kamera yang sebenarnya - rekaman video juga akan berfungsi!

Jika Anda ingin menggunakan ForesAI sebagai paket, Anda dapat memulai dengan menjalankan webcam_benchmark_demo.py dari webcam Anda untuk melihat cara menggunakan API deteksi kamera. Anda juga dapat mencoba video_demo agar inferensi objek dijalankan pada file video pilihan Anda. Untuk konfigurasi lain, silakan lihat skrip * _demo.py bersama dengan file konfigurasi JSON masing-masing untuk cara menggunakan perangkat keras kamera Anda sendiri. Jika menggunakan model Anda sendiri, Anda harus mengubah konfigurasi json dalam folder "demo_configs".

# Benchmarks
Ini adalah tolok ukur terbaik yang saya dapatkan berdasarkan rata-rata di atas aliran 1 menit. Tolok ukur presisi berasal dari laporan oleh penulis khusus mereka. Ini sangat ** sangat mungkin bahwa semua ini dapat ditingkatkan dengan peretasan berbasis model tertentu. Ada banyak pekerjaan baik yang dilakukan dengan SSD-Mobilenet [di sini] (https://github.com/GustavZ/realtime_object_detection).

**Jetson TX2; jetson_clocks enabled; Resolution 480x480**

|Object Detection Models|Frames per Second| CPU % | Combined RAM (MB) | COCO mAP |
|:---------------------:|:---------------:|:-----:|:-----------------:|:--------:|
|SSD-Mobilenet v1 (TensorFlow)|10.01|64.38|1838|21|
|SSD-Mobilenet v1 (TensorFlow, GPU/CPU Split)|18.02|54.89|1799|21|
|SSD-Mobilenet v1 (Movidius)*|10.08|10|247|Not Reported|

|Segmentation Models|Frames per Second| CPU % | Combined RAM (MB) | Mean IoU |
|:-----------------:|:---------------:|:-----:|:-----------------:|:--------:|
|Enet (PyTorch)***|13.81|39.48|2469|60.4|
|ERFnet|7.54|23.54|2464|69.8|
|ResNet 18-8**|3.40|13.89|2297|N/A|
|ResNet 34-8**|1.85|13.26|2296|N/A|

*Measurement less accurate due to not using system tools instead of benchmarking module

**Both ResNet 18 and Resnet 34 requires changing the upsampling algorithm from bilinear interpolation to nearest neighbor for the models to run on the TX2, which will have a negative impact original reported mean IOU 

***Third party implementation

**Nvidia GeForce GTX 1070 8GB GDDR5; i7 4-core 4.20 GHz; 16 GB RAM; Resolution 480x480**

|Object Detection Models|Frames per Second| GPU RAM (MB) | CPU % | RAM (MB) | COCO mAP |
|:---------------------:|:---------------:|:------------:|:-----:|:--------:|:--------:|
|SSD-Mobilenet v1 (TensorFlow)|32.17|363|40.25|1612|21|
|SSD-Mobilenet v1 (TensorFlow, GPU/CPU Split)|61.97|363|58.09|1612|21|
|SSD-Mobilenet v1 (Movidius)|8.51|0|6|57|Not Reported|
|SSD-Mobilenet v2 (TensorFlow)|53.96|2491|35.94|1838|22|
|Mask-RCNN Inception v2|15.86|6573|22.54|1950|25|
|YOLOv3-320|18.91|1413|15.2|688|28.2|

|Segmentation Models|Frames per Second| GPU RAM (MB) | CPU % | RAM (MB) | Mean IoU |
|:-----------------:|:---------------:|:------------:|:-----:|:--------:|:--------:|
|Enet (PyTorch)***|96.4|535|51.35|2185|60.4|
|ERFnet*|63.38|549|40.01|2181|69.8|
|ResNet 18-8*|38.85|605|31.07|2023|60.0|
|ResNet 34-8*|21.12|713|23.29|2020|69.1|
|MobilenetV2-DeeplabV3|4.14|1387|17.49|1530|70.71|

*For some reason running python in the virtual environment for TensorFlow decreased CPU usage by 20%(!). Need to figure out why...

**Measurement less accurate due to not using system tools instead of benchmarking module

***Third party implementation

# To-Dos
Saat ini saya hanya akan fokus pada fitur yang saya perlukan untuk proyek saya dalam waktu dekat, tetapi saya ingin mendengar dari Anda tentang bagaimana membuat perpustakaan ini berguna dalam alur kerja Anda sendiri!

- Dokumentasi
- Buat kerangka yang dapat digeneralisasikan untuk model khusus di Tensorflow dan PyTorch (Model loader)
- Antarmuka untuk mengirim deteksi (misalnya Penerbit independen dari ROS)
- Memungkinkan pengguna untuk mengimplementasikan peretasan manual, model-spesifik
- Standarisasi visualisasi untuk setiap tugas
- Dukungan multi-stick untuk movidus
- Tambahkan pelacak objek
- Integrasi ROS
- Pemantauan penggunaan GPU Nvidia Tegra (Jika di platform Jetson, Anda cukup menggunakan tegrastats.sh)
- Pemantauan penggunaan GPU Nvidia NVML (bisa juga menggunakan nividia-smi)


# Citations
## Code
Many thanks to all the sources cited below. Please feel free to contact me if you have any questions/concerns:

- [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) 

- Stream related classes: [imutils](https://github.com/jrosebr1/imutils) 

- Mobilenet-related hacks that greatly improved speed from [realtime_object_detection](https://github.com/GustavZ/realtime_object_detection)

- [ERFNet implementation and helper functions](https://github.com/Eromera/erfnet_pytorch)

- [Image Segmentation and Object Detection in Pytorch](https://github.com/warmspringwinds/pytorch-segmentation-detection)

- [ENet-PyTorch](https://github.com/bermanmaxim/Enet-PyTorch)

## Models
"Speed/accuracy trade-offs for modern convolutional object detectors."
Huang J, Rathod V, Sun C, Zhu M, Korattikara A, Fathi A, Fischer I, Wojna Z,
Song Y, Guadarrama S, Murphy K, CVPR 2017

"Efficient ConvNet for Real-time Semantic Segmentation", E. Romera, J. M. Alvarez, L. M. Bergasa and R. Arroyo, IEEE Intelligent Vehicles Symposium (IV), pp. 1789-1794, Redondo Beach (California, USA), June 2017. [Best Student Paper Award], [pdf](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17iv.pdf)

"ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation", E. Romera, J. M. Alvarez, L. M. Bergasa and R. Arroyo, Transactions on Intelligent Transportation Systems (T-ITS), December 2017. [pdf](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17tits.pdf)

@article{pakhomov2017deep,
  title={Deep Residual Learning for Instrument Segmentation in Robotic Surgery},
  author={Pakhomov, Daniil and Premachandran, Vittal and Allan, Max and Azizian, Mahdi and Navab, Nassir},
  journal={arXiv preprint arXiv:1703.08580},
  year={2017}
}

Warung Data Indonesia (https://www.warungdata.com)


