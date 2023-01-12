import cv2
import os
from flask import Flask,request,render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

# Konfigurasi flask app
app = Flask(__name__)


#-------------------------- Proses Face Recognition ----------------------------#
# 1) Face Detection: Temukan wajah dalam gambar
# 2) Data Gathering: Analisis bentuk wajah
# 3) Data Comparison: Bandingkan dengan wajah lainnya
# 4) Face Recognition: Buat prediksi siapa yang ada di gambar tsb, misal Tanzil


# Simpan format tanggal
datetoday = date.today().strftime('%e %B %Y') # misal 4 November 2022
datetoday2 = date.today().strftime('%d-%m-%y') # misal 04-11-22


# Inisiasi objek VideoCapture untuk mengakses WebCam
face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml') # model ML Face Detection
cap = cv2.VideoCapture(0)


# Buat direktori jika tidak ada dalam folder Prizen
if not os.path.isdir('histori'):
    os.makedirs('histori')
if not os.path.isdir('static/dataset-wajah'):
    os.makedirs('static/dataset-wajah')
if f'absen-{datetoday2}.csv' not in os.listdir('histori'):
    with open(f'histori/absen-{datetoday2}.csv','w') as f:
        f.write('Name,NIP,Time')


# Total yang sudah registrasi
def totalreg():
    total = os.listdir('static/dataset-wajah')
    return len(total)


# Total yang sudah absensi
def totalattend():
    hasil = pd.read_csv(f'histori/absen-{datetoday2}.csv')
    return len(hasil)


# Identifikasi wajah menggunakan model Machine Learning
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl') # model ML Face Recognition
    return model.predict(facearray)


# Function yang akan mentraining model ke semua wajah yang ada dalam folder dataset-wajah
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/dataset-wajah')
    for user in userlist:
        for imgname in os.listdir(f'static/dataset-wajah/{user}'):
            img = cv2.imread(f'static/dataset-wajah/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5) # semakin besar nilai k maka semakin banyak tetangga yang digunakan untuk proses klasifikasi dan kemungkinan untuk terjadinya noise juga atau akurasinya jadi turun
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')


# Ekstrak info dari folder histori
def extract_attendance():
    df = pd.read_csv(f'histori/absen-{datetoday2}.csv')
    name = df['Name']
    nip = df['NIP']
    time = df['Time']
    l = len(df)
    return name,nip,time,l


# Tambah attendance pada spesifik user
def add_attendance(name):
    nama_user = name.split('_')[0]
    nip_user = name.split('_')[1]
    current_time = datetime.now().strftime('%H:%M:%S') # jam:menit:detik
    
    df = pd.read_csv(f'histori/absen-{datetoday2}.csv')
    if int(nip_user) not in list(df['NIP']): # ternyata typo disini yalord, bkn Time tp NIP
        with open(f'histori/absen-{datetoday2}.csv','a') as f: # disimpan dalam file csv
            f.write(f'\n{nama_user},{nip_user},{current_time}')


#---------------------------------------------- Routing Function -------------------------------------------------#

# Main Index
@app.route('/')
def index():
    name,nip,time,l = extract_attendance()    
    return render_template('index.html',name=name,nip=nip,time=time,l=l,totalreg=totalreg(),datetoday=datetoday) 

# Function ini akan jalan saat klik tombol "Take Attendance Today"
@app.route('/absensi', methods=['GET'])
def absensi():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('index.html',totalreg=totalreg(),datetoday=datetoday,message='Tidak ada pegawai yang terdaftar dalam sistem') 

    cap = cv2.VideoCapture(0)
    ret = True

    while ret:
        ret,frame = cap.read()
        frame = cv2.flip(frame,1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x, y),(x+w, y+h),(19, 57, 189),2) # 2nya tuh line thickness
            # warnanya knp jadi (19, 57, 189) = #1339BD = warna biru ??
            # ternyata itu BGR bkn RGB
            face = cv2.resize(frame[y:y+h,x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1,-1))[0]
            add_attendance(identified_person)
            cv2.rectangle(frame,(x-1,y-23),(x+(w+1), y),(19, 57, 189),cv2.FILLED) # x-1 & w+1 = biar sejajar sm kotak, krn gk bisa pake line thickness
            cv2.putText(frame,f'{identified_person}',(x+6,y-6),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.7,(255, 255, 255),1,cv2.LINE_AA)
        cv2.imshow('Take Attendance',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    name,nip,time,l = extract_attendance()    
    return render_template('index.html',name=name,nip=nip,time=time,l=l,totalreg=totalreg(),datetoday=datetoday,message='Anda berhasil melakukan absen!') 


# Registration Page
@app.route('/registrasi')
def registrasi():
    return render_template('registrasi.html',totalreg=totalreg()) # pantesan totalregnya gk muncul di registrasi, blm dipanggil 

@app.route('/tambah_user', methods=['GET', 'POST'])
def tambah_user():
    nama_pegawai = request.form['nama_pegawai']
    nip_pegawai = request.form['nip_pegawai']
    folder_training = 'static/dataset-wajah/'+nama_pegawai+'_'+str(nip_pegawai)
    if not os.path.isdir(folder_training):
        os.makedirs(folder_training)
    cap = cv2.VideoCapture(0)
    i,j = 0,0

    while 1:
        _,frame = cap.read()
        frame = cv2.flip(frame,1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x, y),(x+w, y+h),(19, 57, 189),2)
            cv2.rectangle(frame,(x-1,y-23),(x+(w+1), y),(19, 57, 189),cv2.FILLED)
            cv2.putText(frame,f'Images Captured: {i}/20',(x+6,y-6),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.7,(255, 255, 255),1,cv2.LINE_AA)
            if j%10==0:
                name = nama_pegawai+'_'+str(i)+'.jpg'
                cv2.imwrite(folder_training+'/'+name,frame[y:y+h,x:x+w])
                i+=1
            j+=1
        if j==200:
            break
        cv2.imshow('Face Training',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    name,nip,time,l = extract_attendance()
    return render_template('registrasi.html',name=name,nip=nip,time=time,l=l,totalreg=totalreg(),datetoday=datetoday,message='Berhasil, Anda terdaftar dalam sistem!') 


# History Page
@app.route('/histori')
def histori():
    name,nip,time,l = extract_attendance()
    return render_template('histori.html',name=name,nip=nip,time=time,l=l,totalreg=totalreg(),datetoday=datetoday,totalattend=totalattend())


if __name__ == '__main__':
    app.run(debug=True)