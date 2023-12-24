import tkinter as tk
from tkinter import filedialog
from ttkthemes import ThemedTk
from PIL import Image, ImageTk

class ImageProcessor:
    def __init__(self, root, image_path, rectangles):
        self.root = root
        self.image_path = image_path
        self.rectangles = rectangles

        self.root.set_theme("alt")
        self.root.title("Rectangle Clicker")

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas_frame = tk.Frame(self.main_frame)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.toolbar_frame = tk.Frame(self.main_frame)
        self.toolbar_frame.pack()

        self.x_scrollbar = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        self.x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 「参照」ボタンを作成し、関連するコマンドを指定
        self.browse_button = tk.Button(self.toolbar_frame, text="参照", command=self.browse_image)
        self.browse_button.pack(side=tk.TOP)

        self.y_scrollbar = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)
        self.y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas = tk.Canvas(self.canvas_frame, xscrollcommand=self.x_scrollbar.set, yscrollcommand=self.y_scrollbar.set, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        


        self.x_scrollbar.config(command=self.canvas.xview)
        self.y_scrollbar.config(command=self.canvas.yview)

        self.canvas.bind("<Configure>", self.on_canvas_configure)

        self.load_image()
        self.draw_rectangles()

        self.canvas.bind("<Button-1>", self.on_click)

    def load_image(self):
        self.image = Image.open(self.image_path)
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.config(scrollregion=(0, 0, self.image.width, self.image.height))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def draw_rectangles(self):
        for rect in self.rectangles:
            x1, y1, x2, y2 = rect
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="red")

    def on_click(self, event):
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        # Check if the click is inside any of the rectangles
        for rect in self.rectangles:
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.process_rectangle(rect)
                break

    def process_rectangle(self, rectangle):
        print("Clicked on rectangle:", rectangle)

    def on_canvas_configure(self, event):
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def browse_image(self):
        new_image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("PNG", "*.png"), ("JPEG", "*.jpg")]
        )
        if new_image_path:
            self.image_path = new_image_path
            self.load_image()

if __name__ == "__main__":
    root = ThemedTk(theme="clam")

    # Use the themed file dialog
    root.tk_setPalette(background='#ececec')

    image_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("PNG", "*.png"), ("JPEG", "*.jpg")]
    )
    if not image_path:
        root.destroy()
    else:
        rectangles = [(50, 50, 150, 150), (200, 100, 300, 200)]  # Example rectangle coordinates
        processor = ImageProcessor(root, image_path, rectangles)
        
        # ウィンドウのサイズを固定
        root.geometry("600x400")  # 例: 幅600ピクセル、高さ400ピクセル
        
        root.mainloop()
