# Deteksi wajah dengan mtcnn pada foto
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN

# Fungsi untuk penghitung jumlah wajah
def wajah_count(gambar, result_list):
	# Variabel data untuk membaca gambar dr file ke array
	data = pyplot.imread(gambar)
	# Menampilkan gambar
	pyplot.imshow(data)
	# Untuk mendapatkan x axis(sumbu pada grafik yang bentuknya mendatar)
	ax = pyplot.gca()
	# Menampilkan crop berbentuk kotak
	for result in result_list:
		# Untuk mendapatkan koordinat objek yang terdeteksi
		x, y, width, height = result['box']
		# Variabel rect untuk membuat crop kotak beserta atributnya
		rect = Rectangle((x, y), width, height, fill=False, color='red')
		# Menggambar/menambahkan crop kotak pada gambar yg dideteksi
		ax.add_patch(rect)
	# Untuk menampilkan informasi
	pyplot.title(f"Jumlah wajah: {len(wajah)}")
	# Menampilkan keseluruhan hasil @matplotlib
	pyplot.show()

# memasukkan file gambar
gambar = 'Filefoto/test4.jpg'
pixels = pyplot.imread(gambar)
# Variabel detektor dengan fungsi Deep Learning mtcnn
detector = MTCNN()

# Mendeteksi wajah pada gambar
wajah = detector.detect_faces(pixels)

# RUN program
wajah_count(gambar, wajah)
