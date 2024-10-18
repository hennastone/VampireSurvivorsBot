from ultralytics import YOLO
import pyautogui
from PIL import Image
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

model = YOLO("weights/best.pt")
model.to("cuda")

x_center = 1920/2
y_center = 1080/2
closest_gem_distance = float('inf')
safe_dist = 200

output_folder = "detections"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def move(boxes, classes):
    topleft = 0
    bottomright = 0
    bottomleft = 0
    topright = 0
    choice = ["topleft",0]
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = box
        x_center_box = (x1 + x2) / 2
        y_center_box = (y1 + y2) / 2
        box_center_coords = (x_center_box, y_center_box)
        enemy_distance = find_distance(x_center_box, y_center_box, x_center, y_center)
        if names[cls].startswith("e_"):
            if enemy_distance<safe_dist:
                runaway(box_center_coords, None)
                continue
            x1, y1, x2, y2 = box
            if x1 < x_center and y1 < y_center:
                topleft += 1
                if choice[1] < topleft:
                    choice[0] = "topleft"
                    choice[1] = topleft
            elif x1 > x_center and y1 < y_center:
                topright += 1
                if choice[1] < topright:
                    choice[0] = "topright"
                    choice[1] = topright
            elif  x1 < x_center and y1 > y_center:
                bottomleft += 1
                if choice[1] < bottomleft:
                    choice[0] = "bottomleft"
                    choice[1] = bottomleft
            else:
                bottomright += 1
                if choice[1] < bottomright:
                    choice[0] = "bottomright"
                    choice[1] = bottomright
    runaway(None, choice[0])
        
def move_gem(coords):
    x, y = coords
    if x < x_center:
        pyautogui.keyUp("d")
        pyautogui.keyDown("a")
    else:
        pyautogui.keyUp("a")
        pyautogui.keyDown("d")
    if y < y_center:
        pyautogui.keyUp("s")
        pyautogui.keyDown("w")
    else:
        pyautogui.keyUp("w")
        pyautogui.keyDown("s")


def runaway(coords=None, choice=None):
    if choice == None:
        x, y = coords
        if x < x_center:
            pyautogui.keyUp("a")
            pyautogui.keyDown("d")
        else:
            pyautogui.keyUp("d")
            pyautogui.keyDown("a")
        if y < y_center:
            pyautogui.keyUp("w")
            pyautogui.keyDown("s")
        else:
            pyautogui.keyUp("s")
            pyautogui.keyDown("w")
    else:
        if choice.startswith("top"):
            pyautogui.keyUp("s")
            pyautogui.keyDown("w")
        else:
            pyautogui.keyUp("w")
            pyautogui.keyDown("s")

        if choice.endswith("left"):
            pyautogui.keyUp("d")
            pyautogui.keyDown("a")
        else:
            pyautogui.keyUp("a")
            pyautogui.keyDown("d")

def find_distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

#screenshot_count = 0

while True:
    screenshot = pyautogui.screenshot()
    screenshot = Image.frombytes("RGB", screenshot.size, screenshot.tobytes())
    result = model.predict(source=screenshot, show=False, conf=0.55)

    """fig, ax = plt.subplots(1)
    ax.imshow(screenshot)"""

    boxes = result[0].boxes.xyxy.tolist()
    classes = result[0].boxes.cls.tolist()
    names = result[0].names
    confidences = result[0].boxes.conf.tolist()

    """for box, cls, conf in zip(boxes, classes, confidences):
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        label = f"{names[cls]} {conf:.2f}"
        plt.text(x1, y1, label, color="white", fontsize=2)

        if names[cls] == "gem":
            gem_distance = find_distance(x_center_box, y_center_box, x_center, y_center)
            if gem_distance < closest_gem_distance:
                closest_gem_distance = gem_distance
                closest_gem_coords = box_center_coords
            move_gem(closest_gem_coords)"""
    
    move(boxes, classes)

    """save_path = os.path.join(output_folder, f"screenshot_{screenshot_count}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    screenshot_count += 1

    plt.close(fig)  """         
            




