import face_recognition
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Compare sketch/photo embeddings")
parser.add_argument("-n1", nargs="?",default=94, help="Index of image 1")
parser.add_argument("-n2",nargs="?",default=94, help = "Index of image 2")
parser.add_argument("-n",nargs="?", help = "Index of image")
args = parser.parse_args()
n1 = str(args.n1)
n2 = str(args.n2)
n = str(args.n)
if n:
    image_1 = face_recognition.load_image_file(f"test_dataset/test_photos/{n}.jpg")
    image_2 = face_recognition.load_image_file(f"test_dataset/test_sketches/{n}.jpg")
elif n1 and n2:
    image_1 = face_recognition.load_image_file(f"test_dataset/test_photos/{n1}.jpg")
    image_2 = face_recognition.load_image_file(f"media/{n2}_generated_realistic_unflattering.jpg")
else :
    print("Please provide either -n for same index or both -n1 and -n2 for different indices.")
    exit(1)

known_encoding = face_recognition.face_encodings(image_1)[0]
unknown_encoding = face_recognition.face_encodings(image_2)[0]

a = np.array(known_encoding)  # length 128
b = np.array(unknown_encoding)  # length 128

cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
distance = np.sum(np.subtract(a,b)**2)
print("Cosine Dsitance: ",cos_sim)
print("Euclidean Distance: ",distance)

results = face_recognition.compare_faces([known_encoding], unknown_encoding)
print(results)