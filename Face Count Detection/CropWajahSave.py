# Mengekstrak dan memplot setiap wajah yang terdeteksi dalam foto
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN

# Fungsi untuk penghitung jumlah wajah
def wajah_count_crop(gambar, result_list):
	# Variabel data untuk membaca gambar dr file ke array
	data = pyplot.imread(gambar)
	# Setiap crop wajah dalam rentang = jumlah wajah yg terdeteksi
	for i in range(len(result_list)):
		# Formula perhitungan
		x1, y1, width, height = result_list[i]['box']
		x2, y2 = x1 + width, y1 + height
		# Membuat hasil masing-masing wajah yang dicrop secara mendatar
		pyplot.subplot(1, len(result_list), i+1)
		pyplot.axis('off')
		# Menampilkan hasilnya
		pyplot.imshow(data[y1:y2, x1:x2])
	# dUntuk menampilkan informasi
	pyplot.title(f"Jumlah wajah: {len(wajah)}")
	# Menampilkan keseluruhan hasil @matplotlib
	pyplot.show()

# Memasukkan file gambar
gambar = 'Filefoto/test6.jpg'
pixels = pyplot.imread(gambar)
# Variabel detektor dengan fungsi Deep Learning mtcnn
detector = MTCNN()

# Mendeteksi wajah pada gambar
wajah = detector.detect_faces(pixels)

# RUN program
wajah_count_crop(gambar, wajah)

