# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 10:58:40 2025

@author: ziyi.liu
"""
import cv2
import numpy as np
from tkinter import Tk, Frame, Button, Label, filedialog, messagebox
from PIL import Image, ImageTk
import os
import matplotlib.pyplot as plt


# 全局變數來儲存當前文件夾路徑、圖片路徑和當前索引
current_folder_path = ""  # 當前文件夾路徑
image_paths = []  # 全部圖片路徑列表
efficient_image_paths = []  # 有效圖片路徑列表
not_efficient_image_paths = []  # 無料圖片路徑列表
color_map_paths = []  # 熱力圖顏色表路徑
current_index = -1  # 當前圖片索引

def update_color_bar(color_bar_label, image_path):
    vertical_image = Image.open(image_path)
    vertical_image.thumbnail((100, 300))  # Scale the image to fit the label
    vertical_photo = ImageTk.PhotoImage(vertical_image)
    color_bar_label.config(image=vertical_photo)
    color_bar_label.image = vertical_photo  # Keep reference to avoid garbage collection
    
def select_source_folder():
    global current_folder_path, image_paths, current_index
    current_folder_path = filedialog.askdirectory(title="選擇資料夾")
    if current_folder_path:
        image_paths = sorted(
            [os.path.join(current_folder_path, f) for f in os.listdir(current_folder_path)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
        )
        if image_paths:
            current_index = 0
            show_image(current_index, current_folder_path)

            # 展示界面其他元素
            image_label.pack(side='left', padx=10)
            processed_image_label.pack(side='left', padx=10)
            prev_button.pack(fill='x', pady=5)
            next_button.pack(fill='x', pady=5)
            reset_button.pack(fill='x', pady=5)
            area_info_label.pack(fill='x', pady=5)
            color_bar_label.pack(side='left', padx=10)

            next_button.config(state='normal')
            prev_button.config(state='normal')
            print("✅ 圖片處理完畢 !")

            if not_efficient_image_paths:
                update_color_bar(color_bar_label, color_map_paths)  # 熱力圖顏色圖
                
                not_efficient_filename = [os.path.basename(path) for path in not_efficient_image_paths]
                message = "以下圖片無效，橡膠片數有缺漏或者過於貼近，照片需要重新拍攝：\n" + "\n".join(not_efficient_filename)
                messagebox.showinfo("警告", message)
        else:
            messagebox.showinfo("警告", "該資料夾沒有圖片。")
            


def show_image(index, folder_path):
    if 0 <= index < len(image_paths):  # 檢查索引是否有效 

        heatmap_rgb = process_images_in_folder(folder_path)   # 執行圖片處理

        display_image(efficient_image_paths[index], image_label)  # 顯示原始圖片        
        display_image(heatmap_rgb, processed_image_label, is_path=False)  # 顯示處理過的圖片
        
        update_area_info(index)  # 更新面積資訊展示    

def display_image(image, label, is_path=True):

    if is_path:
        # 如果 image 是路径，则使用 Image.open 加载图片
        img = Image.open(image)
    else:
        # 如果 image 是数组，则使用 Image.fromarray 将其转为 Image 对象
        img = Image.fromarray(image)

    img.thumbnail((600, 600))  # 调整图片大小，维持比例
    img_tk = ImageTk.PhotoImage(img)  # 将图片转为 Tkinter 识别的形式
    label.config(image=img_tk)  # 更新 Label 窗口中的图片
    label.image = img_tk  # 保留引用避免被垃圾回收
    return img if is_path else image  # 返回图片对象

def update_area_info(index):
    # 更新Label中顯示的面積資訊
    area_info_label.config(text=f"圖片數量 : {len(image_paths)}\n 有效圖片數量 : {len(image_paths) - len(not_efficient_image_paths)}\n\n {index + 1} / {len(image_paths) - len(not_efficient_image_paths)}")

def next_image():
    global current_index
    if current_index != -1 and current_index < len(image_paths) - 1:  # 若還有下一張
        current_index += 1  # 索引加一
        if 0 <= current_index < len(efficient_image_paths):  # 檢查索引是否有效  # 顯示下一張            
            display_image(efficient_image_paths[current_index], image_label)  # 顯示原始圖片
            
            output_folder = os.path.join(current_folder_path, 'pic', 'final', 'heatmap_output')
            output_heatmap_path = os.path.join(output_folder, 'heatmap.png')       
            display_image(output_heatmap_path, processed_image_label, is_path=True)  # 顯示處理過的圖片
            
            update_area_info(current_index)
        else:
            current_index = len(efficient_image_paths) - 1
            
def prev_image():
    global current_index
    if current_index > 0:  # 若還有上一張
        current_index -= 1  # 索引減一
        if 0 <= current_index < len(efficient_image_paths):  # 檢查索引是否有效  # 顯示上一張            
            display_image(efficient_image_paths[current_index], image_label)  # 顯示原始圖片
            
            output_folder = os.path.join(current_folder_path, 'pic', 'final', 'heatmap_output')
            output_heatmap_path = os.path.join(output_folder, 'heatmap.png')       
            display_image(output_heatmap_path, processed_image_label, is_path=True)  # 顯示處理過的圖片
            
            update_area_info(current_index)
        else:
            current_index = 0

def process_images_in_folder(folder_path):
    image_paths = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))
    ]
    image_paths.sort()

    for image_path in image_paths:
        original_img = Image.open(image_path)
        original_img.thumbnail((800, 800))
        process_single_image(original_img, image_path, folder_path)
        
    # 計算熱力圖並儲存圖片
    heatmap_rgb = generate_heatmap(folder_path, output_filename='heatmap.png')
    
    return heatmap_rgb

def process_single_image(original_img, image_path, folder_path):
    global efficient_image_paths, not_efficient_image_paths
    saved_any = False
    original_img_cv = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
    gray_img = cv2.cvtColor(original_img_cv, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closing)

    areas_with_labels = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, num_labels)]
    sorted_regions = sorted(areas_with_labels, key=lambda x: x[0], reverse=True)

    top_n = 4
    masks = []

    if len(sorted_regions) == 4:
        for rank in range(1, top_n):
            if rank < len(sorted_regions):
                area, label = sorted_regions[rank]
                region_mask = np.zeros_like(closing)
                region_mask[labels == label] = 255
                masks.append(region_mask)
        efficient_image_paths.append(image_path)

    else:
        not_efficient_filename = os.path.basename(image_path)
        print(f"❌ 橡膠片數有缺漏或者過於貼近，照片需要重新拍攝，問題圖片檔名：{not_efficient_filename}")
        not_efficient_image_paths.append(image_path)
        return  # 直接跳出，不做後續儲存或合併


    cut_folder_paths = []
    img_paths = []     # 儲存路徑

    for i, mask in enumerate(masks):
        masked_image_bgr = cv2.bitwise_and(original_img_cv, original_img_cv, mask=mask)
        cut_folder = os.path.join(os.path.dirname(image_path), f"pic/cut/{i+1}")
        
        cut_folder = os.path.abspath(os.path.join(cut_folder))
        cut_folder_paths.append(cut_folder)
        os.makedirs(cut_folder, exist_ok=True)
        
        filename = os.path.basename(image_path)
        output_filename = f"proceed_{filename}"
        output_path = os.path.abspath(os.path.join(cut_folder, output_filename))
        
        output_folder = os.path.join(cut_folder, 'aligned_images')
        os.makedirs(output_folder, exist_ok=True)
        img_paths.append(os.path.abspath(os.path.join(output_folder, output_filename)))
        masked_image_RGB = cv2.cvtColor(masked_image_bgr, cv2.COLOR_BGR2RGB)
        try:
            pil_image = Image.fromarray(masked_image_RGB)
            pil_image.save(output_path)
            saved_any = True
        except Exception as e:
            print(f"❌ 圖片儲存失敗：{e}")
            
            
    for path in cut_folder_paths:        
        align_and_save_images(path) # 處理平移
            
    
    # 使用 PIL 讀取、轉為 RGB，再轉 OpenCV 的 BGR 格式  (將切割過後平移過後的小片合併) 
    cv_images = []
    for path in img_paths:
        pil_img = Image.open(path).convert('RGB')
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv_images.append(cv_img)

    # 合併圖像（保留每個像素的最大值）
    combined_cv = cv_images[0]
    for img in cv_images[1:]:
        combined_cv = np.maximum(combined_cv, img)

    # 轉回 RGB，再轉回 PIL
    combined_rgb = cv2.cvtColor(combined_cv, cv2.COLOR_BGR2RGB)
    combined_pil = Image.fromarray(combined_rgb) 
    
    # 新的final資料夾路徑
    final_folder = os.path.join(folder_path, 'pic', 'final')
    
    # 如果final資料夾不存在就創建
    os.makedirs(final_folder, exist_ok=True)
    
    # 取得原始檔名（不含副檔名）
    basename = os.path.splitext(os.path.basename(image_path))[0]
    
    # 建立完整的儲存路徑
    save_path = os.path.join(final_folder, f'{basename}_final.jpg')
    
    # 儲存合併後圖像
    combined_pil.save(save_path)

    if saved_any:
        print("✅ 圖片儲存成功 !")
    else:
        print("❌ 圖片儲存失敗 !")
        
def get_contour_bounds(image):
    """取得最大輪廓的邊界與輪廓本身"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(c), c
    return None, None

def align_and_save_images(input_folder):
    """處理所有圖像，對齊後儲存到輸出資料夾"""
    if not os.path.exists(input_folder):
        print(f"找不到輸入資料夾: {input_folder}")
        return

    output_folder = os.path.join(input_folder, 'aligned_images')
    os.makedirs(output_folder, exist_ok=True)

    reference_bounds = None
    reference_position = None

    for filename in os.listdir(input_folder):
        if not (filename.lower().endswith('.jpg') or filename.lower().endswith('.png')):
            continue

        img_path = os.path.join(input_folder, filename)
        pil_img = Image.open(img_path).convert('RGB')
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        bounds, contour = get_contour_bounds(img)

        if bounds:
            x, y, w, h = bounds
            # print(f"圖片 {filename} 的邊界: 左上角({x}, {y}), 寬 {w}, 高 {h}")

            if reference_bounds is None:
                reference_bounds = bounds
                reference_position = (x + w // 2, y + h // 2)
                # print(f"✅ 設定參考圖片：{filename}")
                # 儲存原始參考圖片
                output_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                Image.fromarray(output_rgb).save(os.path.join(output_folder, filename))
            else:
                current_position = (x + w // 2, y + h // 2)
                dx = reference_position[0] - current_position[0]
                dy = reference_position[1] - current_position[1]

                M = np.float32([[1, 0, dx], [0, 1, dy]])
                aligned_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                output_rgb = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)
                output_path = os.path.join(output_folder, filename)
                Image.fromarray(output_rgb).save(output_path)
                # print(f"💾 已儲存對齊圖片：{output_path}")
        else:
            print(f"⚠️ 未找到有效輪廓：{filename}")
            
def generate_heatmap(input_folder, output_filename='heatmap.png'):
    global color_map_paths
    # Settings for heatmap generation
    lower = np.array([15, 65, 30])
    upper = np.array([40, 255, 160])
    
    # Prepare output folder
    output_folder = os.path.join(input_folder, 'pic', 'final', 'heatmap_output')
    input_folder = os.path.join(input_folder, 'pic', 'final')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Read the first image to get dimensions
    first_img_name = next((f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))), None)
    if not first_img_name:
        raise ValueError("No image files found in the input folder.")
    
    first_img_path = os.path.join(input_folder, first_img_name)
    pil_img = Image.open(first_img_path).convert('RGB')
    first_img = np.array(pil_img)
    first_img = cv2.cvtColor(first_img, cv2.COLOR_RGB2BGR)
    h, w, _ = first_img.shape
    count_matrix = np.zeros((h, w), dtype=np.uint32)
    
    # Process each image
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png')):
            img_path = os.path.join(input_folder, filename)
            pil_img = Image.open(img_path).convert('RGB')
            img = np.array(pil_img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Convert to HSV
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Create mask
            mask = cv2.inRange(hsv_img, lower, upper)
            # Update count matrix
            count_matrix += mask // 255
    
    # Normalize count matrix
    count_matrix = count_matrix.astype(np.float32)
    normalized_image = cv2.normalize(count_matrix, None, 0, 255, cv2.NORM_MINMAX)
    
    # Create heatmap
    heatmap = cv2.applyColorMap(normalized_image.astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # 创建一个条件掩码，找到所有等于 (0, 0, 128) 的像素 (將熱力圖的藍色背景轉成黑色)
    mask = np.all(heatmap_rgb == [0, 0, 128], axis=-1)

    # 将这些像素值改为黑色
    heatmap_rgb[mask] = [0, 0, 0]
    
    # 儲存熱力圖
    output_heatmap_path = os.path.join(output_folder, output_filename)
    Image.fromarray(heatmap_rgb).save(output_heatmap_path)
    
    # 創建熱力圖顏色表
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)  # 为垂直颜色条设计

    # 创建一个图和轴，设置图形大小
    fig, ax = plt.subplots(figsize=(2, 6))
    
    # 在轴中显示渐变
    ax.imshow(gradient, aspect='auto', cmap='jet')
    
    # 隐藏坐标轴
    ax.axis('off')
    
    # 调整布局，以控制颜色条的宽度
    plt.subplots_adjust(left=0.6, right=0.8)

    # 手动添加0%至100%的标签
    # 由于 y 轴高度从 0 到 256，因而需要相应调整文本的位置
    positions = [0.01, 0.25, 0.5, 0.75, 0.99]
    labels = ['0%', '25%', '50%', '75%', '100%']

    for pos, label in zip(positions, labels):
        # 根据位置设定文本，因y值从0到256，因此乘以256
        ax.text(0.6, pos * 256, label, va='center', ha='left', fontsize=12)
        
    color_map_paths = os.path.join(output_folder, 'color_map.png')
        
    # 保存颜色条到文件
    plt.savefig(color_map_paths, bbox_inches='tight')
    
    return heatmap_rgb

def reset():
    global current_folder_path, image_paths, efficient_image_paths, not_efficient_image_paths, current_index
    current_folder_path = ""
    image_paths.clear()
    efficient_image_paths.clear()
    not_efficient_image_paths.clear()
    current_index = -1

    # 清除图片和信息显示
    image_label.pack_forget()
    processed_image_label.pack_forget()
    prev_button.pack_forget()
    next_button.pack_forget()
    reset_button.pack_forget()
    area_info_label.pack_forget()
    color_bar_label.pack_forget()

    # 只显示选择资料夹按钮
    source_button.pack(fill='x', pady=10)
    
    print("重置到初始狀態")


if __name__ == '__main__':
    app = Tk()
    app.title('影像辨識 v1.0')
    app.geometry('1200x700')

    frame = Frame(app)
    frame.pack(padx=20, pady=20)

    # 初始只显示选择资料夹按钮
    source_button = Button(frame, text="選擇資料夾", command=select_source_folder)
    source_button.pack(fill='x', pady=10)

    # 创建其他控件，但不显示
    image_label = Label(frame)
    processed_image_label = Label(frame)
    prev_button = Button(frame, text="上一張", command=prev_image, state='disabled')
    next_button = Button(frame, text="下一張", command=next_image, state='disabled')
    reset_button = Button(frame, text="重置", command=reset)
    area_info_label = Label(frame, text="", font=("calibri", 12))
    color_bar_label = Label(frame)

    app.mainloop()