import tkinter as tk
from concurrent.futures import ThreadPoolExecutor

import cv2
import keyboard
import mss
import numpy as np
import pyautogui

with_cuda = bool(int(input("Использовать CUDA? (0 - нет, 1 - да):\n")))
print(f"CUDA: {with_cuda}")


class SelectionWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Select Area")
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-alpha', 0.3)
        self.canvas = tk.Canvas(self.root, bg='gray', cursor='cross', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)

        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.selection = None

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def on_click(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y,
                                                    outline="red", width=2)

    def on_drag(self, event):
        cur_x = event.x
        cur_y = event.y
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, cur_x, cur_y)

    def on_release(self, event):
        end_x = event.x
        end_y = event.y
        self.selection = (self.start_x, self.start_y, end_x - self.start_x, end_y - self.start_y)
        self.root.destroy()

    def get_selection(self):
        self.root.mainloop()
        return self.selection


def capture_screen(region=None):
    with mss.mss() as sct:
        if region:
            left, top, width, height = region
            right = left + width
            bottom = top + height
            screen = sct.grab((left, top, right, bottom))
        else:
            screen = sct.grab(sct.monitors[1])  # или другой монитор по умолчанию
        screen_np = np.array(screen)
        return cv2.cvtColor(screen_np, cv2.COLOR_BGRA2GRAY)


def find_template(screen_gray, template_gray, threshold):
    if with_cuda:
        screen_cuda = cv2.cuda.GpuMat()
        template_cuda = cv2.cuda.GpuMat()

        screen_cuda.upload(screen_gray)
        template_cuda.upload(template_gray)

        matcher = cv2.cuda.createTemplateMatching(cv2.CV_8UC1, cv2.TM_CCOEFF_NORMED)
        result_cuda = matcher.match(screen_cuda, template_cuda)
        result_cuda = result_cuda.download()

        min_val, max_val, min_loc, max_loc = cv2.cuda.minMaxLoc(result_cuda)
    else:
        result = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        return max_loc, max_val
    else:
        return None, 0


# Загрузка изображений
template_path = 'target3.png'
template_img = cv2.imread(template_path)
template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

template2_path = 'target.png'
template2_img = cv2.imread(template2_path)
template2_gray = cv2.cvtColor(template2_img, cv2.COLOR_BGR2GRAY)

# Создание окна для выделения области
selection_window = SelectionWindow()
region = selection_window.get_selection()


def process_screen_and_template(region, template_gray, template2_gray, threshold=0.8):
    screen_img = capture_screen(region)
    location, confidence = find_template(screen_img, template_gray, threshold=threshold)
    if location is None:
        location, confidence = find_template(screen_img, template2_gray, threshold=threshold)
    return location, confidence, screen_img


executor = ThreadPoolExecutor(max_workers=4)

while True:
    if keyboard.is_pressed('q'):
        break

    future = executor.submit(process_screen_and_template, region, template_gray, template2_gray)

    if keyboard.is_pressed('alt') or keyboard.is_pressed('ctrl'):
        continue

    location, confidence, screen_img = future.result()

    if confidence > 0:
        absolute_location = (region[0] + location[0], region[1] + location[1])
        h, w, _ = template_img.shape
        center_x = absolute_location[0] + w // 2
        center_y = absolute_location[1] + h // 2

        pyautogui.click(center_x, center_y)

    if keyboard.is_pressed('alt'):
        if confidence > 0:
            h, w, _ = template_img.shape
            top_left = location
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(screen_img, top_left, bottom_right, (0, 255, 0), 2)
            cv2.imshow('Detected Template', screen_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
