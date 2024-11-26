# **Digital Skill Fair 33.0 - Data Science Class**

<p align='center'>
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/></a>&nbsp;&nbsp;
    <a href="https://jupyter.org/">
        <img src="https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white"/></a>&nbsp;&nbsp;
    <a href="https://streamlit.io/">
        <img src="https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white"/></a>&nbsp;&nbsp;

</p>

<div align="center">

| Profile       |                                           |
| ------------- | ----------------------------------------- |
| Nama          | Mawan/Nur Muhammad Himawan                |
| Class         | Data Science Class                        |
| Progam        | Digital Skill Fair 33.0                   |


</div>


<div align="center">
<figure>
    <img src ="app/assets/demo_application.png" alt="Demo App">
    <figcaption align="center"><b>Demo Application</b></figcaption>
</figure>
</div>

## **Demo Project**

[Breast Cancer Application](https://dsf-breast-cancer.streamlit.app/)


## Domain Proyek

Domain yang akan dibahas pada proyek machine learning ini adalah **Kesehatan** dengan judul **"Deteksi Kanker Payudara Menggunakan Algoritma Machine Learning"**.

### Latar Belakang

Kanker payudara adalah salah satu jenis kanker yang paling umum dan menjadi penyebab utama kematian pada perempuan di seluruh dunia. Menurut American Cancer Society, pada tahun 2023 terdapat estimasi 297.790 kasus baru dan 43.170 kasus kematian akibat penyakit ini. Penyakit ini terjadi ketika sel-sel di payudara tumbuh tidak terkendali. Pada tahap awal, sering kali tidak ada gejala yang jelas, namun pada tahap lanjut dapat muncul benjolan, nyeri, perubahan bentuk payudara, dan perubahan kulit. Jika tidak ditangani dini, kanker ini dapat menyebar ke organ lain dan berisiko fatal.

Deteksi dini melalui pemeriksaan rutin seperti mamografi, USG payudara, atau pemeriksaan klinis sangat penting untuk meningkatkan peluang kesembuhan. Namun, tantangan seperti kurangnya akses fasilitas kesehatan atau rendahnya kesadaran masyarakat menjadi kendala utama. Oleh karena itu, sistem deteksi berbasis klasifikasi data medis menggunakan machine learning sangat diperlukan. Sistem ini dapat menganalisis data biopsi untuk membantu diagnosis yang lebih cepat dan pengobatan yang lebih efektif.

Referensi:

- [2024—First Year the US Expects More than 2M New Cases of Cancer](https://www.cancer.org/research/acs-research-news/facts-and-figures-2024.html) 
- [Kanker Masih Membebani Dunia](https://sehatnegeriku.kemkes.go.id/baca/blog/20240506/3045408/kanker-masih-membebani-dunia/)


### Solusi Machine Learning

Deteksi kanker payudara sering kali mengandalkan analisis citra medis yang kompleks seperti mamografi, USG payudara, atau pemeriksaan klinis. Namun, metode ini bisa memakan waktu dan memerlukan interpretasi klinis yang kompleks. Dengan menggunakan model machine learning, diharapkan dapat memberikan penilaian yang lebih cepat dan akurat. Model-model sederhana seperti SVM, Random Forest hingga MLP, mampu memproses data input untuk menghasilkan prediksi apakah kanker bersifat malignant atau benign. Sistem ini dapat digunakan sebagai alat bantu diagnostik dalam pengambilan keputusan medis yang lebih cepat, yang pada gilirannya dapat meningkatkan hasil perawatan dan penanganan pada kanker payudara.


### Metode Pengolahan
Dataset terdiri dari 31 kolom, di mana 30 kolom digunakan sebagai fitur dan satu kolom digunakan sebagai label klasifikasi (diagnosis). Fitur-fitur dalam dataset adalah numerik dan mencakup berbagai ukuran geometri tumor seperti radius, tekstur, perimeter, dan area, dengan tipe data float64. Kolom label memiliki dua jenis diagnosis, malignant dan benign. Proses pre-processing mencakup pembagian dataset menjadi 70% untuk pelatihan dan 30% untuk testing. Fitur numerik dinormalisasi menggunakan metode standarosaso untuk memastikan bahwa semua fitur memiliki skala yang seragam, sehingga model dapat melatih lebih efektif dan mengurangi bias yang disebabkan oleh perbedaan skala antar fitur.


### Arsitektur Model
Pada fase ini, saya menggunakan tiga algoritma klasifikasi yang berbeda, yaitu Support Vector Machine (SVM), Random Forest (RF), dan Multilayer Perceptron (MLP), untuk mendeteksi kanker payudara.

- **Support Vector Machine (SVM)**, bekerja untuk menemukan hyperplane atau pemisah yang dapat memaksimalkan jarak (margin) antar kelas dalam ruang n-dimensi untuk mengklasifikasikan titik-titik data. Memaksimalkan jarak margin akan memberikan kejelasan terkait klasifikasi kelas sehingga titik data yang baru dilihat dapat diklasifikasikan dengan lebih baik.

  <p align='center'>
      <img src ="https://github.com/nurmuhimawann/MLT_Dicoding/blob/main/images/SVM.png?raw=true" alt="svm">
  </p>


  Pada tahap modelling, [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) yang dipakai menggunakan metode kernel dan menerima semua vektor input yang diberikan pada data training dengan menerapkan parameter `rbf` yang dipakai sebagai kernel tricks nya. Kernel ini dikenal memiliki performa yang baik dan hasil dari pelatihan memiliki nilai error yang kecil. fungsi kernel rbf adalah sebagai berikut `K(x,xi) = exp(-gamma \* sum((x – xi^2))`

  - Kelebihan

    Mampu bekerja dengan baik pada data yang relatif sedikit.

    Pengklasifikasian SVM dapat memberikan model dengan akurasi tinggi dan bekerja dengan baik dengan ruang dimensi tinggi.

  - Kekurangan

    Sulit diaplikasikan pada data yang sangat besar karena memiliki waktu pelatihan yang tinggi.

- **Random Forest (RF)**, salah satu algoritma machine learning terbaik yang digunakan dalam klasifikasi dalam jumlah data yang besar. Random Forest memakai pendekatan kombinasi dari beberapa pohon keputusan (decision tree) yang datanya akan dipilih secara random. Dalam random forest, penentuan klasifikasi dilakukan berdasarkan hasil voting dari tree yang terbentuk. sehingga, pemakaian jumlah tree yang lebih banyak dapat menghasilkan tingkat akurasi yang lebih optimal. Tree yang dihasilkan oleh random forest dilatih menggunakan metode bagging. Bagging akan bekerja dengan memilih fitur secara random dengan menerapkan sampling with replacement. Kemudian, dari hasil ini akan diperoleh model tree klasifikasi. proses ini akan terus berulang hingga mendapatkan jumlah tree (k) yang diinginkan. kemudian dari jumlah tree yang ada, masing-masing tree akan memberikan hasil prediksi. langkah terakhir, proses *majority voting* akan dilakukan untuk menentukan prediksi akhir. Pada proyek machine learning ini, implementasi random forest akan dilakukan dengan memakai modul [RandomForestClassifier()](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) yang telah tersedia pada library scikit-learn. parameter `n_estimator` dipakai untuk menentukan jumlah tree. disini saya memakai 100 tree. Kemudian setelah menentukan parameter model, proses selanjutnya adalah building model dan prediksi yang dilakukan menggunakan data testing. hasil dari testing akan dievaluasi menggunakan metriks accuracy.

  <p align='center'>
      <img src="https://github.com/nurmuhimawann/MLT_Dicoding/blob/main/images/random-forest.png?raw=true" height=300px alt="random-forest">
  </p>
  
  - Kelebihan :
    
    Random Forest bekerja sangat baik pada data dengan jumlah yang sangat besar.
    
    Hasil pembelajaran yang diperoleh pada random forest memiliki tingkat akurasi yang sangat baik.
    
    Random Forest dapat memberikan perkiraan variabel yang penting dalam proses klasifikasi.
    
  - Kekurangan :
    
    Untuk type data kategorikal, random forest tidak bisa bekerja dengan optimal dan cenderung menghasilkan hasil prediksi yang bias.
    
    Waktu runtime yang lama karena random forest menggunakan data dalam jumlah yang besar dan random tree yang banyak pula.
    

- **Multilayer Perceptron (MLP)**, jenis jaringan saraf tiruan yang bekerja dengan menyusun lapisan-lapisan neuron untuk mempelajari hubungan kompleks dalam data. MLP memiliki lapisan input, satu atau lebih lapisan tersembunyi, dan lapisan output, yang memungkinkan model mempelajari pola non-linear melalui aktivasi fungsi seperti ReLU atau sigmoid.

  - Kelebihan:
    
    Kemampuan Generalisasi: Bekerja baik dengan data non-linear yang kompleks.
    
    Fleksibilitas: Dapat disesuaikan dengan jumlah lapisan dan neuron.
    
    Performansi Tinggi: Cocok untuk data dengan dimensi yang besar.

  - Kekurangan:
    
    Waktu Latihan Lama: Terutama jika arsitektur terlalu kompleks.
    
    Overfitting: Jika tidak menggunakan regularisasi yang tepat.
    
    Kebutuhan Komputasi Tinggi: Membutuhkan sumber daya besar untuk dataset yang besar.

Penggunaan kombinasi model ini bertujuan untuk membandingkan performa masing-masing arsitektur dalam mendeteksi risiko kanker payudara, berdasarkan metrik seperti akurasi, presisi, recall, dan F1-score pada confusion matrix.


### **Metrik Evaluasi**
Berdasarkan evaluasi model, SVM menunjukkan performa terbaik dengan akurasi tertinggi sebesar 98%, diikuti oleh MLP dengan akurasi 97%, dan Random Forest dengan akurasi 94%. Hasil ini mengindikasikan bahwa SVM memiliki kemampuan generalisasi paling baik untuk dataset yang digunakan, diikuti oleh MLP yang hampir setara. Meskipun Random Forest memiliki akurasi lebih rendah, model ini tetap cukup andal dan dapat menjadi alternatif. Secara keseluruhan, semua model menunjukkan hasil yang memuaskan dalam mendeteksi kanker payudara.


## **Future Work**

-   [x] Data Acquisition **(Done)**
-   [ ] Exploratory Data Analysis & Visualization
-   [ ] Feature Engineering
-   [x] Modelling **(Done)**
-   [x] Model Evaluation **(Done)**
-   [ ] Hyperparameters Tuning
-   [ ] Trying Other Algorithms - For now, just SVM, RF & MLP.
-   [ ] Model Explainability – Implement SHAP or LIME for better understanding of predictions.
-   [ ] Feature Importance Analysis – Identify key features in the model.
-   [ ] Model Documentation – Provide detailed documentation on the running app, training process and model.


## **Resources**

-   **UCI Machine Learning Repository** : https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
