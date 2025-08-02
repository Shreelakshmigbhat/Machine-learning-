import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
image_size = (64, 64)
cat_folder = r"C:\Users\Shreelakshmi G Bhat\Downloads\working (2)\working\Supervised and Unsupervised learning\supervised\classification\animals\dog"
dog_folder = r"C:\Users\Shreelakshmi G Bhat\Downloads\working (2)\working\Supervised and Unsupervised learning\supervised\classification\animals\cat"
def load_images_from_folder(folder, label):
    features, labels = [], []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, image_size)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            flat = gray.flatten()
            features.append(flat)
            labels.append(label)
    return features, labels
cat_features, cat_labels = load_images_from_folder(cat_folder, 0)
dog_features, dog_labels = load_images_from_folder(dog_folder, 1)
X = np.array(cat_features + dog_features)
y = np.array(cat_labels + dog_labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
print("✅ Model Trained. Accuracy:", accuracy_score(y_test, svm.predict(X_test)))
class ImagePredictorApp:
    def __init__(self, master):
        self.master = master
        master.title("Cat vs Dog Classifier")
        master.geometry("500x600")
        master.configure(bg="#f7f7f7")
        self.label = tk.Label(master, text="Choose an image to predict:", font=("Arial", 14), bg="#f7f7f7")
        self.label.pack(pady=10)
        self.canvas = tk.Canvas(master, width=256, height=256, bg="white", bd=2, relief="ridge")
        self.canvas.pack(pady=20)
        self.button = tk.Button(master, text="Browse Image", command=self.load_image, font=("Arial", 12), bg="#6c63ff", fg="white")
        self.button.pack(pady=10)
        self.result_label = tk.Label(master, text="", font=("Arial", 16, "bold"), bg="#f7f7f7")
        self.result_label.pack(pady=20)
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
        try:
            pil_img = Image.open(file_path).resize((256, 256))
            self.tk_img = ImageTk.PhotoImage(pil_img)
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
            img = cv2.imread(file_path)
            img = cv2.resize(img, image_size)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            flat = gray.flatten().reshape(1, -1)
            prediction = svm.predict(flat)[0]
            label = "Cat" if prediction == 0 else "Dog"
            self.result_label.config(text=f"Prediction: {label}", fg="#333")
        except Exception as e:
            messagebox.showerror("Error", f"Could not process image.\n{str(e)}")
if __name__ == "__main__":
    root = tk.Tk()
    app = ImagePredictorApp(root)
    root.mainloop()
