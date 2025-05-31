![top_image_small2](https://github.com/user-attachments/assets/ef3a3e48-1136-4012-99f2-6cd36a69b199)

# JetCamCounter

JetCamCounter は、NVIDIA Jetson 上で動作するリアルタイム物体検出＆カウントシステムです。YOLOv4-tiny を使用して、カメラまたは動画ファイルから人や車などの物体を検出・トラッキングし、通過数を自動でカウントします。

交通量調査をAIで代替することを想定しています。他の用途（入退場管理、人数把握、防犯、店舗分析など）にも応用可能です。

![person2AI2](https://github.com/user-attachments/assets/d2ac3db8-9a73-4508-bd0b-80800ccd0235)

<br/>

## 📹 動作デモ

https://github.com/user-attachments/assets/a51f9eac-a9ab-4847-a90a-51b164dc3cf2

- 中央の縦線を横切ったオブジェクトをカウントし、カウント数を左上に表示
- カメラ映像・動画ファイルの両方に対応
- 結果は動画ファイルおよびテキストファイルに出力し、後から確認可能

<br/>

## 🚀 背景と目的

交通量調査はこれまで人手で行うことが一般的でしたが、以下の課題がありました。

- 長時間にわたる作業が必要
- 人的ミスが発生する可能性がある
- データのデジタル化に手間がかかる
- 人を雇うことによりコストが多くかかる

JetCamCounter では、Jetson のエッジAI性能と YOLOv4-tiny の軽量性を活かし、**カメラ映像をリアルタイムに処理**して自動的にカウントする仕組みを構築。省電力で小型な Jetson により、**屋外でも長時間の稼働が可能**です。また、人が張り付いてカウントする必要がないため、人手にかかるコストを削減できます。

<br/>

## 🔍 特長

- ✅ Jetson 対応・低電力で高効率
- ✅ YOLOv4-tiny による高速な検出
- ✅ カメラまたは動画ファイルを入力に切り替え可能
- ✅ トラッキングにより通過数をカウント
- ✅ 任意のオブジェクトクラス（車・人など）に対応

<br/>

## 🧠 活用例

- 🚗 **交通量調査**（交差点の車両カウント）
- 🏢 **入退場管理**（オフィス・施設での人数把握）
- 🛍 **小売店舗分析**（来店客の動向分析）
- 🧍‍♂️ **行列の長さ推定**（待機人数のモニタリング）
- 🔐 **防犯用途**（不審者の侵入検出）

<br/>

## ⚙️ セットアップ手順（Jetson用）

### 1. Jetson 初期セットアップ

以下のページなどを参考に、Jetsonを初期セットアップします。  
[Getting Started with AI on Jetson Nano](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-RX-02+V2)

### 2. Jetson 用 PyTorch のインストール
以下のページに従い、Jetson 用 PyTorch をインストールします。  
[PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)


### 3. Python 依存ライブラリのインストール
```bash
sudo apt install python3-opencv
pip3 install numpy
```

### 4. darknet のビルド（YOLOv4-tiny）
```bash
git clone https://github.com/AlexeyAB/darknet.git
cd darknet
# Makefile を編集（GPU=1, CUDNN=1, OPENCV=1 などを有効化）
sed -i 's/GPU=0/GPU=1/' Makefile
sed -i 's/CUDNN=0/CUDNN=1/' Makefile
sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
sed -i 's/OPENCV=0/OPENCV=1/' Makefile
sed -i 's/LIBSO=0/LIBSO=1/' Makefile
sed -i 's/^ARCH=.*/ARCH= -gencode arch=compute_53,code=[sm_53,compute_53]/' Makefile
# make
make -j$(nproc)
```

### 5. 重み・設定ファイルのダウンロード
```bash
# cfg ファイルと学習済み重みをダウンロード
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg -P cfg/
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/coco.data -P cfg/
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
```

### 6. main.py を darknet フォルダにコピー
main.py を darknet フォルダにコピーします。

<br/>

## ▶️ 実行方法
以下のコマンドで JetCamCounter を実行できます。

### カメラ入力で実行
```bash
python3 main.py
```

### 動画ファイル入力で実行
```bash
python3 main.py input_video.mp4
```

実行時に生成されるファイル:  
- 出力動画: result/<base_name>_<mode>_<タイムスタンプ>.mp4  
- カウントログ: result/<base_name>_<mode>_count_log_<タイムスタンプ>.txt

<br/>

## 💡 今後の展望
- 結果ファイルをサーバへ自動アップロード
- 夜間・赤外線カメラ対応
- 継続的な検知精度向上
