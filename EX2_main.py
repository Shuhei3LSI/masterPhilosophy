#参考文献「https://ni4muraano.hatenablog.com/entry/2017/08/10/101053」
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from numpy.lib import math
from EX2_unet import UNet
import matplotlib.pyplot as plt
from PIL import Image

#ホログラムのパラメータ
OUT_SIZE = 256
IMAGE_X = OUT_SIZE
IMAGE_Y = OUT_SIZE
DATA_X = IMAGE_X * 2
DATA_Y = IMAGE_Y * 2
PITCH_X = 10.0e-6
PITCH_Y = 10.0e-6
lamda = 520e-9
z = 0.1

#学習のパラメータ
IMAGE_SIZE = OUT_SIZE           # 画像のサイズ
NUM_EPOCH = 20                  # エポック数
BATCH_SIZE = 2                  # バッチサイズ        
first_layer_filter_count = 64   # 一番初めのConvolutionフィルタ枚数は64
input_channel_count = 2         # 入力２チャンネル（amp,phase）
output_channel_count = 2        # 出力２チャンネル（amp,phase）

#データセットの形:
    # 間引き率 = 1/2

# 保存パス
path_out = 'data/'

# 環境パターン
env = 'b'

# 保存パス(データセット)
path_dataset = 'dataset/' + env + '/'

# パラメータの保存
def parameters():
    f = open(path_out + 'environment=' + env + '_cgh_parameters.txt', 'w')
    f.write("複素ホログラム " + "画素数:" + str(IMAGE_X) + " ピッチ:" + str(PITCH_X) + " 波長:" + str(lamda) + " 伝搬距離:"+ str(z))
    f.close()

# 角スペクトル伝搬(ゼロパティングあり) complex f_xy[IMAGE_Y][IMAGE_Y](戻り値も同様型),伝搬距離distance
def calculation(f_xy, distance):
    # ゼロパティング
    img_a = np.zeros((DATA_X, DATA_Y),dtype=np.complex128)
    for n in range(DATA_Y):
        for m in range(DATA_X):
            if n < DATA_Y / 4 or DATA_Y * 3 / 4 <= n or m < DATA_X / 4 or DATA_X * 3 / 4 <= m:
                g = 0    
            else:
                img_a[n][m] = f_xy[int(n - DATA_Y / 4)][int(m - DATA_Y / 4)]
    # FFT
    f_uv = np.fft.fft2(img_a)
    # 規格化
    f_uv /= (DATA_X * DATA_Y)
    #Hの計算
    H = np.zeros((DATA_X,DATA_Y),dtype=np.complex128)
    for n in range(DATA_Y):
        for m in range(DATA_X):
            du = 1 / (PITCH_X * DATA_X)
            dv = 1 / (PITCH_Y * DATA_Y)
            H[n][m] = complex(np.cos(2 * np.pi * distance * np.sqrt(pow(1 / lamda, 2) - pow(du * (m - DATA_X / 2), 2) - pow(dv * (n - DATA_Y / 2), 2)))
            ,np.sin(2 * np.pi * distance * np.sqrt(pow(1 / lamda, 2) - pow(du * (m - DATA_X / 2), 2) - pow(dv * (n - DATA_Y / 2), 2))))
    # 画像の中心に低周波数の成分がくるように並べかえる
    shifted_h = np.fft.fftshift(H)
    H = shifted_h
    # 角スペクトル計算
    OUT = np.zeros((DATA_X,DATA_Y),dtype=np.complex128)
    OUT = f_uv * H
    # IFFT
    OUT_ifft = np.fft.ifft2(OUT)
    # ゼロパティングの解除
    complex_cgh = np.zeros((IMAGE_X, IMAGE_Y),dtype=np.complex128)
    for n in range(DATA_Y):
        for m in range(DATA_X):
            if n < DATA_Y / 4 or DATA_Y * 3 / 4 <= n or m < DATA_X / 4 or DATA_X * 3 / 4 <= m:
                g = 0    
            else:
                complex_cgh[int(n - DATA_Y / 4)][int(m - DATA_Y / 4)] = OUT_ifft[n][m]
    
    return complex_cgh

# 画像化256gray complex pic[IMAGE_Y][IMAGE_Y] flag = 実部:0,虚部:1,振幅:2,位相:3
def save_pic(cpx,path,flag):
    # 保存配列の用意
    pic = np.zeros((IMAGE_X,IMAGE_Y),np.float32)
    
    # 複素振幅の何を保存か選択
    if flag == 0:
        for n in range(IMAGE_Y):
            for m in range(IMAGE_Y):
                pic[n][m] = cpx[n][m].real
    if flag == 1:
        for n in range(IMAGE_Y):
            for m in range(IMAGE_Y):
                pic[n][m] = cpx[n][m].imag
    if flag == 2:
        for n in range(IMAGE_Y):
            for m in range(IMAGE_Y):
                pic[n][m] = abs(cpx[n][m])
    if flag == 3:
        for n in range(IMAGE_Y):
            for m in range(IMAGE_Y):
                pic[n][m] = math.atan2(cpx[n][m].imag, cpx[n][m].real)

    # 再生像の保存
    max = 0
    min = pic[0][0]
    for n in range(IMAGE_Y):
        for m in range(IMAGE_Y):
            pic[n][m] = pic[n][m]
            if(pic[n][m] > max):
                max = pic[n][m]
            if(pic[n][m] < min):
                min = pic[n][m]
    
    for n in range(IMAGE_Y):
        for m in range(IMAGE_X):
            pic[n][m] = 255 * (pic[n][m] - min) / (max - min)

    pil_img = Image.fromarray(pic.astype(np.uint8))
    pil_img.save(path)

# CGHの再生 complex npy[IMAGE_X][IMAGE_X]
def rec_cgh(npy,save_path):
    # 入力の格納と伝搬計算
    complex_cgh = np.zeros((IMAGE_X, IMAGE_Y),dtype=np.complex128)
    complex_cgh = calculation(npy, -z)
    
    # 可視に変換
    result = np.zeros((IMAGE_X,IMAGE_Y),np.float32)
    for n in range(IMAGE_Y):
        for m in range(IMAGE_Y):
            result[n][m] = np.sqrt(complex_cgh[n][m].real ** 2 + complex_cgh[n][m].imag ** 2)
            #result[n][m] = complex_cgh[n][m].real ** 2 + complex_cgh[n][m].imag ** 2
    
    # 再生像の保存
    max = 0
    min = result[0][0]
    for n in range(IMAGE_Y):
        for m in range(IMAGE_Y):
            if(result[n][m] > max):
                max = result[n][m]
            if(result[n][m] < min):
                min = result[n][m]
    for n in range(IMAGE_Y):
        for m in range(IMAGE_X):
            result[n][m] = 255 * (result[n][m] - min) / (max - min)
    pil_img = Image.fromarray(result.astype(np.uint8))
    pil_img.save(save_path) 

# 実部と虚部の規格化 complex npy[IMAGE_X][IMAGE_X] 実部:-1~+1 虚部:-1~+1
def normalize(npy):
    max_re = 0
    min_re = npy[0][0].real
    max_im = 0
    min_im = npy[0][0].imag
    for n in range(IMAGE_Y):
        for m in range(IMAGE_X):
            if(npy[n][m].real > max_re):
                max_re = npy[n][m].real
            if(npy[n][m].real < min_re):
                min_re = npy[n][m].real
            if(npy[n][m].imag > max_im):
                max_im = npy[n][m].imag
            if(npy[n][m].imag < min_im):
                min_im = npy[n][m].imag
    for n in range(IMAGE_Y):
        for m in range(IMAGE_X):
            npy[n][m] = complex((npy[n][m].real - min_re) / (max_re - min_re) * 2 - 1,(npy[n][m].imag - min_im) / (max_im - min_im) * 2 - 1)

# 線形補完 complex npy[IMAGE_X][IMAGE_X]
def complement(npy):
    # 前処理
    complex_cgh = np.zeros((DATA_X, DATA_Y),dtype=np.complex128)
    for n in range(DATA_Y):
        for m in range(DATA_X):
            if n < DATA_Y / 4 or DATA_Y * 3 / 4 <= n or m < DATA_X / 4 or DATA_X * 3 / 4 <= m:
                g = 0    
            else:
                complex_cgh[n][m] = npy[int(n - DATA_Y / 4)][int(m - DATA_Y / 4)]
    
    # 線形補完
    re = np.zeros((DATA_X,DATA_Y),np.float32)
    im = np.zeros((DATA_X,DATA_Y),np.float32)
    for n in range(int(IMAGE_Y / 2),int(IMAGE_Y / 2 + IMAGE_Y)):
        for m in range(int(IMAGE_X / 2),int(IMAGE_X / 2 + IMAGE_X)):
            count = 0
            if n % 2 == 0 and m % 2 == 0:
                dd = 0
            else:#補完場所の認識
                if complex_cgh[n + 1][m].real == 0 and complex_cgh[n + 1][m].imag == 0:
                    g = 0    
                else:
                    count += 1
                    re[n][m] += complex_cgh[n + 1][m].real
                    im[n][m] += complex_cgh[n + 1][m].imag

                if complex_cgh[n + 1][m + 1].real == 0 and complex_cgh[n + 1][m + 1].imag == 0:
                    g = 0    
                else:
                    count += 1
                    re[n][m] += complex_cgh[n + 1][m + 1].real
                    im[n][m] += complex_cgh[n + 1][m + 1].imag

                if complex_cgh[n][m + 1].real == 0 and complex_cgh[n][m + 1].imag == 0:
                    g = 0    
                else:
                    count += 1
                    re[n][m] += complex_cgh[n][m + 1].real
                    im[n][m] += complex_cgh[n][m + 1].imag

                if complex_cgh[n - 1][m + 1].real == 0 and complex_cgh[n - 1][m + 1].imag == 0:
                    g = 0    
                else:
                    count += 1
                    re[n][m] += complex_cgh[n - 1][m + 1].real
                    im[n][m] += complex_cgh[n - 1][m + 1].imag

                if complex_cgh[n - 1][m].real == 0 and complex_cgh[n - 1][m].imag == 0:
                    g = 0    
                else:
                    count += 1
                    re[n][m] += complex_cgh[n - 1][m].real
                    im[n][m] += complex_cgh[n - 1][m].imag

                if complex_cgh[n - 1][m - 1].real == 0 and complex_cgh[n - 1][m - 1].imag == 0:
                    g = 0    
                else:
                    count += 1
                    re[n][m] += complex_cgh[n - 1][m - 1].real
                    im[n][m] += complex_cgh[n - 1][m - 1].imag

                if complex_cgh[n][m - 1].real == 0 and complex_cgh[n][m - 1].imag == 0:
                    g = 0    
                else:
                    count += 1
                    re[n][m] += complex_cgh[n][m - 1].real
                    im[n][m] += complex_cgh[n][m - 1].imag

                if complex_cgh[n + 1][m - 1].real == 0 and complex_cgh[n + 1][m - 1].imag == 0:
                    g = 0    
                else:
                    count += 1
                    re[n][m] += complex_cgh[n + 1][m - 1].real
                    im[n][m] += complex_cgh[n + 1][m - 1].imag

                complex_cgh[n][m] = complex(re[n][m] / count,im[n][m] / count)
                

    # ゼロパティングの解除
    comp = np.zeros((IMAGE_X, IMAGE_Y),dtype=np.complex128)
    for n in range(DATA_Y):
        for m in range(DATA_X):
            if n < DATA_Y / 4 or DATA_Y * 3 / 4 <= n or m < DATA_X / 4 or DATA_X * 3 / 4 <= m:
                g = 0    
            else:
                comp[int(n - DATA_Y / 4)][int(m - DATA_Y / 4)] = complex_cgh[n][m]
    
    print(comp)

    # CGHの保存(位相)　flag = 実部:0,虚部:1,振幅:2,位相:3
    save_pic(comp,path_out + "comp_cgh_phase.bmp",3)     
    # 再生
    rec_cgh(comp,path_out + 'comp_rec.bmp')  

# npy保存 complex npy[IMAGE_Y][IMAGE_X][2]で実部、虚部を保存
def save_npy(npy,path):
    npy_a = np.zeros((IMAGE_X,IMAGE_Y,2),np.float32)
    for n in range(IMAGE_Y):
        for m in range(IMAGE_X):
            npy_a[n][m][0] = npy[n][m].real
            npy_a[n][m][1] = npy[n][m].imag
    np.save(path, npy_a)

# 学習用データセット(half)の生成
def dataset_generate(NUM):
        for n in range(0,NUM):
            # パスの設定
            path_in = 'C:/Users/metar/Desktop/U-net/CWO_resize_gray_tr8/data_cwo_pet/' + str(n) + '.bmp'
            path_dataset_cgh = path_dataset + 'CGH/' + str(n)
            path_dataset_zero = path_dataset  + 'ZERO/' + str(n)
            # 画像を読み込む
            img = Image.open(path_in)
            # グレースケールに変換
            gray_img = img.convert('L')
            # (IMAGE_X,IMAGE_Y)に変換
            resize_img = gray_img.resize((IMAGE_X,IMAGE_Y))
            # NumPy 配列に変換
            f_xy = np.array(resize_img)
            # 再生時に向けてルート
            f_xy = np.sqrt(f_xy)
            # 伝搬計算
            complex_cgh = np.zeros((IMAGE_X, IMAGE_Y),dtype=np.complex128)
            complex_cgh = calculation(f_xy, z)
            #実部:-1~+1 虚部:-1~+1で規格化
            normalize(complex_cgh)
            # npyで実部、虚部を[IMAGE_Y][IMAGE_X][2]で保存 float32で約500KB
            save_npy(complex_cgh,path_dataset_cgh)
            # 1/2間引き
            for n in range(IMAGE_Y):
                for m in range(IMAGE_X):
                    if (m + n) % 2 == 0:
                        dd = 0
                    else:
                        complex_cgh[n][m] = complex(0,0)
            # npyで実部、虚部を[IMAGE_Y][IMAGE_X][2]で保存 float32で約500KB
            save_npy(complex_cgh,path_dataset_zero)

#既存教師データセットから入力データセット(three quarters)を生成
def gen_indata_from_kyousi(NUM):
    for n in range(0,NUM):
        # パスの設定
        path_in_cgh = path_dataset + 'CGH/' + str(n) + ".npy"
        path_out = path_dataset + 'three_quarters/' + str(n)
        npy = np.load(path_in_cgh)
        npy_a = np.zeros((IMAGE_X,IMAGE_Y,2),np.float32)
        for n in range(IMAGE_Y):
            for m in range(IMAGE_X):
                if(n % 2 == 0 and m % 2 == 0):
                    npy_a[n][m][0] = npy[n][m][0]
                    npy_a[n][m][1] = npy[n][m][1]

        np.save(path_out, npy_a)

# テスト用データ(three quarters)の生成
def test_data():
    # 画像を読み込む
    img = Image.open(path_out + 'testdatapic/' + '7008.bmp')
    # グレースケールに変換
    gray_img = img.convert('L')
    # (IMAGE_X,IMAGE_Y)に変換
    resize_img = gray_img.resize((IMAGE_X,IMAGE_Y))
    # NumPy 配列に変換
    f_xy = np.array(resize_img)
    # 再生時に向けてルート
    f_xy = np.sqrt(f_xy)
    # 伝搬計算
    complex_cgh = np.zeros((IMAGE_X, IMAGE_Y),dtype=np.complex128)
    complex_cgh = calculation(f_xy, z)
    #実部:-1~+1 虚部:-1~+1で規格化
    normalize(complex_cgh)
    # CGHの画像保存(位相)　flag = 実部:0,虚部:1,振幅:2,位相:3
    save_pic(complex_cgh,path_out + "orig_cgh_phase.bmp",3)     
    # オリジナルの再生
    rec_cgh(complex_cgh,path_out + 'orig_rec.bmp')

    # 実部と虚部を元のthree quartersに変換間引き
    for n in range(IMAGE_Y):
        for m in range(IMAGE_X):
            if n % 2 == 0 and m % 2 == 0:
                dd = 0
            else:
                complex_cgh[n][m] = complex(0,0)

    # CGHの保存(位相)　flag = 実部:0,虚部:1,振幅:2,位相:3
    save_pic(complex_cgh,path_out + "covr_cgh_phase.bmp",3)     
    # 再生
    rec_cgh(complex_cgh,path_out + 'convr_rec.bmp')  

    # npyで保存 float32で約500KB
    save_npy(complex_cgh,path_out + 'test_in/' + 'convr_npy')

    # 線形補完 位相、再生像の保存
    complement(complex_cgh)

# NN読み込み用
def load_npz(folder_path):
    import os
    image_files = os.listdir(folder_path)
    image_files.sort()
    images = np.zeros((len(image_files),IMAGE_SIZE,IMAGE_SIZE,2), np.float32)
    for i, image_file in enumerate(image_files):
        image = np.load(folder_path + os.sep + image_file)
        images[i] = image
    return images, image_files
    
# 学習曲線の保存
def save_curve(history,path):
    fig = plt.figure(figsize=(10, 5))  # グラフを表示するスペースを用意
    plt.plot(range(1, NUM_EPOCH+1), history.history['loss'], "-o")
    plt.plot(range(1, NUM_EPOCH+1), history.history['val_loss'], "-o")
    plt.title('model loss')
    plt.ylabel('loss')  # Y軸ラベル
    plt.xlabel('epoch')  # X軸ラベル
    plt.grid()
    plt.legend(['検証データ', '学習データ'], loc='best')
    fig.savefig(path)

# npy[IMAGE_Y][IMAGE_X][2]をcomplex[IMAGE_Y][IMAGE_X]に変換して戻す。（#振幅:-1~1を0~1に変換）
def to_complex(npy,complex_a):
    for n in range(IMAGE_Y):
        for m in range(IMAGE_X):
            complex_a[n][m] = complex(npy[n][m][0],npy[n][m][1])

# U-Netのトレーニングを実行する関数
def train_unet(): 
    # datasetの読み込み
    X_train, file_names = load_npz(path_dataset + 'three_quarters')
    #X_train, file_names = load_npz(path_dataset + 'ZERO')
    Y_train, file_names_y = load_npz(path_dataset + 'CGH')

    # U-Netの生成
    network = UNet(input_channel_count, output_channel_count, first_layer_filter_count)
    model = network.get_model()
    model.compile(loss='mse', optimizer=Adam())

    # 学習の実行
    history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCH, verbose=1, validation_split=0.1)
    
    # 学習曲線の保存
    save_curve(history,path_out + 'environment=' + env + "_epoch=" + str(NUM_EPOCH) + '_learning curve.png')
    
    # 重みの保存
    model.save_weights(path_out + 'environment=' + env + "_epoch=" + str(NUM_EPOCH) + '_weights.hdf5')

# 学習後のU-Netによる予測を行う関数
def predict():
    # テストデータの読み込み
    X_test, file_names = load_npz(path_out + 'test_in')

    # NNの構築
    network = UNet(input_channel_count, output_channel_count, first_layer_filter_count)
    model = network.get_model()
    
    # 重みのロード
    model.load_weights(path_out + 'environment=' + env + "_epoch=" + str(NUM_EPOCH) + '_weights.hdf5')

    # NNに入力
    Y_pred = model.predict(X_test, BATCH_SIZE)

    # 結果を再生像にして保存 y[IMAGE_Y][IMAGE_X][2] 実部、虚部が-1~1
    for i, y in enumerate(Y_pred):
        # complexに変換
        complex_y = np.zeros((IMAGE_Y, IMAGE_X),dtype=np.complex128)
        to_complex(y,complex_y)
        
        # CGHの保存(位相)　flag = 実部:0,虚部:1,振幅:2,位相:3
        save_pic(complex_y,path_out + "train_out_cgh_phase_" + str(i) + ".bmp",3)     
        # 再生
        rec_cgh(complex_y,path_out + "train_out_cgh_rec_" + str(i) + '.bmp')  

# PSNR
def psnr(img_1, img_2, data_range=255):
    mse_t = np.mean((img_1.astype(float) - img_2.astype(float)) ** 2)
    return 10 * np.log10((data_range ** 2) / mse_t)

# 固定記述パスで画質評価(PSNR only)
def Image_quality_evaluation():
    path_orgi = path_out + 'orig_rec.bmp'
    path_mise = path_out + 'convr_rec.bmp'
    path_trin = path_out + "train_out_cgh_rec_" + str(0) + '.bmp'
    img = Image.open(path_orgi)
    gray_img = img.convert('L')
    resize_img = gray_img.resize((IMAGE_X,IMAGE_Y))
    img_orig = np.array(resize_img)

    img = Image.open(path_mise)
    gray_img = img.convert('L')
    resize_img = gray_img.resize((IMAGE_X,IMAGE_Y))
    img_mise = np.array(resize_img)

    img = Image.open(path_trin)
    gray_img = img.convert('L')
    resize_img = gray_img.resize((IMAGE_X,IMAGE_Y))
    img_trin= np.array(resize_img)

    print("PSNR : " + str(psnr(img_orig,img_mise)))
    print("PSNR : " + str(psnr(img_orig,img_trin)))

def main():
    NUM = 7000
    parameters()

    #dataset_generate(NUM)

    #gen_indata_from_kyousi(NUM)

    #test_data()

    train_unet()

    #predict()

    #Image_quality_evaluation()


if __name__ == '__main__':
    aa = 0
    ss = 1

    main()



