import tkinter as tk
from tkinter import filedialog
from ttkthemes import ThemedTk
from PIL import Image, ImageTk

class ImageProcessor:
    def __init__(self, root, image_path, rectangles):
        self.root = root
        self.image_path = image_path
        self.rectangles = rectangles
        
        root.option_add("*font", ["IPAゴシック", 12])

        self.root.set_theme("alt")
        self.root.title("Change Label")

        # ウインドウ全体のメインフレーム
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 画像を表示するキャンバスのフレーム
        self.canvas_frame = tk.Frame(self.main_frame)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 縦軸のスクロールバー
        self.x_scrollbar = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        self.x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 横軸のスクロールバー
        self.y_scrollbar = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)
        self.y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 参照ボタン等を配置するツールバーのフレーム
        self.toolbar_frame = tk.Frame(self.main_frame)
        self.toolbar_frame.pack()
        
        # 画像を選択するファイルダイアログを開く「参照」ボタン
        self.browse_button = tk.Button(self.toolbar_frame, text="参照", command=self.browse_image)
        self.browse_button.pack(side=tk.TOP)

        # キャンバスの描画
        self.canvas = tk.Canvas(self.canvas_frame, xscrollcommand=self.x_scrollbar.set, yscrollcommand=self.y_scrollbar.set, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.x_scrollbar.config(command=self.canvas.xview)
        self.y_scrollbar.config(command=self.canvas.yview)
        
        # マウスイベント
        self.canvas.bind("<Configure>", self.on_canvas_configure) # 
        self.canvas.bind("<Button-1>", self.left_click) # 左クリック（領域の選択）
        self.canvas.bind("<B3-Motion>", self.right_drag) # 右ドラッグ（画像の移動）
        self.canvas.bind("<ButtonRelease-3>", self.right_release)  # マウスが離れたときに座標を初期化
        
        # # 作成予定
        # self.canvas.bind("<Control-Key-a>", self.save_labels) # ラベル変更結果の保存

        self.load_image()
        self.draw_rectangles()

    # 画像読み込み
    def load_image(self):
        self.image = Image.open(self.image_path)
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.config(scrollregion=(0, 0, self.image.width, self.image.height))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def draw_rectangles(self):
        for rect in self.rectangles:
            x1, y1, x2, y2 = rect
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="red")

    # 左クリックの動作（領域選択）
    def left_click(self, event):
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        # Check if the click is inside any of the rectangles
        for rect in self.rectangles:
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.process_rectangle(rect)
                break
    
    # 右ドラッグの動作（画像移動）
    def right_drag(self, event):
        cursor_x, cursor_y = event.x, event.y

        # 前回の座標が保存されていない場合は初期化
        if hasattr(self, "prev_x") and hasattr(self, "prev_y"):

            # マウスの移動距離を計算
            delta_x = (cursor_x - self.prev_x)
            delta_y = (cursor_y - self.prev_y)

            # 画像を移動する
            self.canvas.xview_scroll(-delta_x, "units")
            self.canvas.yview_scroll(-delta_y, "units")

            # 現在の座標を保存
            self.prev_x = cursor_x
            self.prev_y = cursor_y
        
        else:
            # 初回の右ドラッグの場合、現在のマウス位置を保存
            self.prev_x, self.prev_y = event.x, event.y

    # マウスが離れたときに座標を初期化
    def right_release(self, event):
        del self.prev_x
        del self.prev_y
        
    # # 
    # def save_labels(self, event):
        
        

    def process_rectangle(self, rectangle):
        print("Clicked on rectangle:", rectangle)

    def on_canvas_configure(self, event):
        self.canvas.config(scrollregion=self.canvas.bbox("all"))


    def browse_image(self):
        new_image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("PNG", "*.png"), ("JPEG", "*.jpeg *jpg")]
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
        filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("PNG", "*.png"), ("JPEG", "*.jpeg *jpg")]
    )
    if not image_path:
        root.destroy()
    else:
        rectangles = [(50, 50, 150, 150), (200, 100, 300, 200)]  # Example rectangle coordinates
        processor = ImageProcessor(root, image_path, rectangles)
        
        # ウィンドウのサイズを固定
        root.geometry("900x600")  # 幅×高さ（ピクセル）
        
        root.mainloop()
