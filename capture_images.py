import pyautogui
import time
import os

screenshots_folder = os.path.join('screenshots')

# screenshots klasörü yoksa oluştur
if not os.path.exists(screenshots_folder):
    os.makedirs(screenshots_folder)
i = 99
# Sürekli olarak 10 saniyede bir ekran görüntüsü al ve kaydet
try:
    while True:
        # Ekran görüntüsünü al
        screenshot = pyautogui.screenshot()

        file_name = f'screenshot_{i}.png'

        # Dosya yolunu oluştur
        file_path = os.path.join(screenshots_folder, file_name)

        # Ekran görüntüsünü kaydet
        screenshot.save(file_path)
        print(f'{i} kaydedildi.')

        # 10 saniye bekle
        time.sleep(10)
        i += 1
except KeyboardInterrupt:
    print('Program durduruldu.')
