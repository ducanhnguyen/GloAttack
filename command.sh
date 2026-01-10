#!/bin/bash

# Hàm tải file với fallback cho nhiều hệ điều hành
download_file() {
    local url=$1
    local output=$2

    if command -v wget &> /dev/null; then
        wget -O "$output" "$url"
    elif command -v curl &> /dev/null; then
        curl -L -o "$output" "$url"
    else
        echo "❌ Cần wget hoặc curl để tải file"
        exit 1
    fi
}

# Tạo thư mục chứa annotation nếu chưa có
mkdir -p annotations

download_file "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "annotations_trainval2017.zip"

unzip annotations_trainval2017.zip -d annotations

git clone https://github.com/ultralytics/yolov5.git

cd yolov5

pip install -r requirements.txt

cd ..
pip install git+https://github.com/cleverhans-lab/cleverhans.git

pip install scikit-learn
pip install pycocotools
pip install scikit-image
pip install piq

pip install transformers

pip install timm

pip install tensorboard