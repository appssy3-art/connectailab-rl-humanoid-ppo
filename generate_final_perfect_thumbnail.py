import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_BOLD = "C:/Windows/Fonts/malgunbd.ttf"

def get_font(size):
    return ImageFont.truetype(FONT_BOLD, size)

def draw_stroke_text(draw, pos, text, font, fill_color, stroke_color=(0, 0, 0), stroke_width=5, anchor="mm"):
    x, y = pos
    draw.text((x, y), text, font=font, fill=fill_color, stroke_width=stroke_width, stroke_fill=stroke_color, anchor=anchor)

def make_perfect_thumbnail():
    print("🚀 [유튜브 쇼츠 완벽 규격 썸네일 생성 시작]...")
    
    # 1. Load native 1080x1920 stride frame (step 28)
    frame = cv2.imread("stride_frame_28.jpg")
    h, w, _ = frame.shape  # 1920, 1080
    
    # Character center_x in step 28 was 752.
    # We want it at x = 540 (exact center of 1080 width).
    # Shift = 540 - 752 = -212 px
    shift_x = -212
    M = np.float32([[1, 0, shift_x], [0, 1, 0]])
    shifted = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # Convert to PIL
    canvas = Image.fromarray(cv2.cvtColor(shifted, cv2.COLOR_BGR2RGB))
    
    # 2. Text Boxes with Precise Margins
    # Top Margin: 180px
    # Top Box Height: 180px (Y: 180 .. 360)
    # Character Head is around Y: 607 (Breathing room = 247px)
    # Character Feet are around Y: 1449
    # Bottom Box Height: 180px (Y: 1530 .. 1710)
    # Bottom Margin: 210px (Safe from YouTube Shorts UI cutoff of 250px and Shelf crop)
    
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    
    box_w = 900
    box_x1 = (w - box_w) // 2
    box_x2 = box_x1 + box_w
    
    # --- Top Box ---
    top_y1, top_y2 = 180, 360
    # Clean black box with subtle orange border (exact vibe of '걸음마 포기하고')
    ov_draw.rounded_rectangle([box_x1, top_y1, box_x2, top_y2], radius=22, 
                               fill=(0, 0, 0, 245), outline=(255, 122, 0, 230), width=4)
    
    # --- Bottom Box ---
    bot_y1, bot_y2 = 1530, 1710
    # Clean black box with subtle white border (exact vibe of '롤러를 선택한 AI')
    ov_draw.rounded_rectangle([box_x1, bot_y1, box_x2, bot_y2], radius=22, 
                               fill=(0, 0, 0, 245), outline=(255, 255, 255, 200), width=4)
    
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    
    # --- Text Rendering ---
    f_title = get_font(94)
    
    # Top Text: '걸음마 시켰더니' (Bright vibrant orange #FF7A00)
    top_text_y = (top_y1 + top_y2) // 2
    draw_stroke_text(draw, (w // 2, top_text_y), "걸음마 시켰더니", f_title, 
                     fill_color=(255, 122, 0), stroke_color=(0, 0, 0), stroke_width=6)
    
    # Bottom Text: '마라토너 된 AI' (Pure clean white #FFFFFF)
    bot_text_y = (bot_y1 + bot_y2) // 2
    draw_stroke_text(draw, (w // 2, bot_text_y), "마라토너 된 AI", f_title, 
                     fill_color=(255, 255, 255), stroke_color=(0, 0, 0), stroke_width=6)
    
    # Save the master final files
    canvas.save("thumbnail_shorts_perfect_safe.png", quality=98)
    canvas.save("thumbnail_shorts_channel_style.png", quality=98)
    canvas.save("thumbnail_shorts_split_9x16.png", quality=98)
    print("Master thumbnail saved to thumbnail_shorts_channel_style.png!")
    
    # Generate Shelf Preview (Simulating YouTube Grid 1:1.55 display)
    crop_shelf_h = 1675
    crop_top = (h - crop_shelf_h) // 2
    shelf_preview = canvas.crop((0, crop_top, w, crop_top + crop_shelf_h))
    shelf_preview.save("preview_youtube_shelf_crop.png")
    print(f"Simulated shelf crop saved: preview_youtube_shelf_crop.png (Top margin: {top_y1 - crop_top}px, Bot margin: {crop_top + crop_shelf_h - bot_y2}px)")

if __name__ == "__main__":
    make_perfect_thumbnail()

