import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from classifier import img_normalization, classify


class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Классификация VGG16")
        self.setFixedSize(700, 350)

        self.btn = QPushButton("Выбрать изображение...", self)
        self.btn.setFixedSize(180, 28)
        self.btn.move(85, 161)
        self.btn.clicked.connect(self.evt_btn_clicked)

        self.img_icon = QLabel(self)
        self.img_icon.setFixedSize(224, 224)
        self.img_icon.move(413, 63)
        self.img_icon.setPixmap(QPixmap(str(Path("images", "img_icon.png"))))

        self.result = QLabel(self)
        self.result.move(85, 200)
        self.result.resize(180, 56)

    def evt_btn_clicked(self):
        res = QFileDialog.getOpenFileName(self, "Выберите файл", f"{os.getcwd}", "Images (*.png *bmp *.jpg)")
        # self.result.setText("Обработка...")
        sqr_image = img_normalization(res[0])
        img_file_name = os.path.basename(res[0])
        img_file_path = str(Path("images", "norm_images", f"norm_{img_file_name}"))
        sqr_image.save(img_file_path)
        self.img_icon.setPixmap(QPixmap(img_file_path))
        self.print_result(sqr_image)

    def print_result(self, img):
        res = classify(img)
        self.result.setText(res)


    


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app_window = AppWindow()
    app_window.show()
    sys.exit(app.exec_())

