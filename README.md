#Kidney Stone detection based on Vision Transformer model

Trong đề tài này, việc phát hiện sỏi thận bằng hình ảnh X-quang được đề xuất với kỹ thuật học sâu (DL), đã đạt được tiến bộ đáng kể trong lĩnh vực trí tuệ nhân tạo. Tổng số 1.799 hình ảnh đã được sử dụng bằng cách chụp các hình ảnh CT cắt ngang khác nhau cho mỗi người. Mô hình tự động được phát triển cho thấy độ chính xác là khá cao khi sử dụng hình ảnh CT để phát hiện sỏi thận.

Các tập dữ liệu và code sẽ được đính kèm - https://github.com/yildirimozal/Kidney_stone_detection, có thể chạy mã code đã cho bằng cách sử dụng [nền tảng google colab](https://colab.research.google.com/)

Train 

    kidney_stone: 625
    normal: 828
Test

    Kidnet_stone: 165
    Normal: 181

#Mô tả đề tài

![Optional Text](../master/TomTatModel.jpg)  
![](.\TomTatModel.jpg)

#YOLOv5
Đầu tiên, chúng ta kết nối với Google Driver:

    from google.colab import drive
    drive.mount('/content/drive')
Clone model và cài đặt các requirements:

    git clone https://github.com/ultralytics/yolov5  # clone
    cd yolov5
    pip install -r requirements.txt  # install
Giải nén tập dữ liệu

    !unzip /content/drive/'My Drive'/LuanVan/data.zip
Tải file /yolov5/data/coco128.yaml về và sửa lại thành file 
        

![Optional Text](../master/image1.jpg)  
![](.\image1.jpg)
Tiến hành train model với custom dataset. Ta chọn pretrained yolov5-s với các thông số phù hợp:

    !python train.py --img 640 --batch 8 --epochs 10 --data /content/drive/MyDrive/LuanVan/yolov5/data/coco128.yaml  --weights weights/yolov5s.pt
Sau khi train, kết quả train sẽ được lưu vào các thư mục runs/train/exp, trọng số (weights) của model Yolov5 sẽ được lưu trong thư mục weights, weights của của epoch tốt nhất và best.pt và epoch cuối cùng last.pt
![](.\ketqua.jpg)   

![Optional Text](../master/ketqua.jpg)  
Phát hiện đối tượng trên ảnh bằng lệnh:

    !python detect.py --source /content/drive/MyDrive/LuanVan/data/images/val --weights /content/drive/MyDrive/LuanVan/yolov5/runs/train/exp2/weights/last.pt  --img 640 --save-txt --save-conf
Kết quả sẽ được lưu vào các thư mục runs/detect/exp
![Optional Text](../master/Kidney_stone30.jpg)  

![](.\Kidney_stone30.jpg)


#Vision Transformer - ViT
Chuẩn bị cơ sở dữ liệu
    
    import glob
    kidney_stone = glob.glob('/content/drive/MyDrive/LuanVan/dataset/Kidney_stone/*.*')
    normal = glob.glob('/content/drive/MyDrive/LuanVan/dataset/Normal/*.*')
    #print(kidney_stone)
    data = []
    labels = []
    
    for i in kidney_stone:   
        image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', target_size= (224,224))
        #print(image)
        image=np.array(image)
        data.append(image)
        labels.append(0)
        
    for i in normal:   
        image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', target_size= (224,224))
        image=np.array(image)
        data.append(image)
        labels.append(1)
    
    data = np.array(data)
    labels = np.array(labels)
    #print(data)
    print(labels)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2,
                                                    random_state=42)

Xác định các tham số

    learning_rate = 0.001
    weight_decay = 0.0001
    batch_size = 256
    num_epochs = 1000
    image_size = 224  # We'll resize input images to this size
    patch_size = 16  # Size of the patches to be extract from the input images
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = 8
    mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier

Tăng cường dữ liệu

    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import tensorflow_addons as tfa
    data_augmentation = keras.Sequential(
        [
            layers.Normalization(),
            layers.Resizing(image_size, image_size),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ],
        name="data_augmentation",
    )
    # Compute the mean and the variance of the training data for normalization.
    #data_augmentation.layers[0].adapt(x_train)
    data_augmentation.layers[0].adapt(X_train)

Xây dựng mạng MLP

    def mlp(x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

Chuyển hình ảnh thành các bản vá

    class Patches(layers.Layer):
        def __init__(self, patch_size):
            super(Patches, self).__init__()
            self.patch_size = patch_size
    
        def call(self, images):
            batch_size = tf.shape(images)[0]
            patches = tf.image.extract_patches(
                images=images,
                sizes=[1, self.patch_size, self.patch_size, 1],
                strides=[1, self.patch_size, self.patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            )
            patch_dims = patches.shape[-1]
            patches = tf.reshape(patches, [batch_size, -1, patch_dims])
            return patches
Đầu ra

![Optional Text](../master/patch_1.jpg)  
![](.\patch_1.jpg)

Kế tiếp sẽ xây dựng các khối cho máy mô hình. Đầu tiên,  sẽ sử dụng dữ liệu tăng cường sẽ đi qua khối trình tạo bản vá và sau đó dữ liệu sẽ đi qua khối mã hóa bản vá. Trong khối Transformer, sử dụng một lớp self-attention trên các chuỗi bản vá. Đầu ra từ khối biến áp sẽ đi qua một đầu phân loại giúp tạo ra các đầu ra cuối cùng. 

    def create_vit_classifier():
        inputs = layers.Input(shape=input_shape)
        # Augment data.
        augmented = data_augmentation(inputs)
        # Create patches.
        patches = Patches(patch_size)(augmented)
        # Encode patches.
        encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
        # Create multiple layers of the Transformer block.
        for _ in range(transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, 
                key_dim=projection_dim, 
                dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])
    
        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        # Add MLP.
        features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
        # Classify outputs.
        logits = layers.Dense(num_classes)(features)
        # Create the Keras model.
        model = keras.Model(inputs=inputs, outputs=logits)
        return model

Biên dịch và huấn luyện
- Biên dịch


    optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model = create_vit_classifier()
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
           keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
           keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"), ],)

- Huấn luyện


    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
    )

Kết quả đánh giá

    results = model.evaluate(X_test, y_test)
    print('Test loss: {:4f}'.format(results[0]))
    print('Test accuracy: {:4f}'.format(results[1]))

![Optional Text](../master/image2.jpg)  
![](.\image2.jpg)
"# LuanVanCNTT" 

