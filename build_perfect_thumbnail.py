import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_BOLD = "C:/Windows/Fonts/malgunbd.ttf"

def get_font(size):
    return ImageFont.truetype(FONT_BOLD, size)

def draw_stroke_text(draw, pos, text, font, fill_color, stroke_color=(0, 0, 0), stroke_width=4, anchor="mm"):
    x, y = pos
    draw.text((x, y), text, font=font, fill=fill_color, stroke_width=stroke_width, stroke_fill=stroke_color, anchor=anchor)

def build_perfect_thumbnail():
    print("🎯 [완벽한 안전마진 & 채널 일체형 썸네일 제작 시작]...")
    
    w, h = 1080, 1920
    canvas = Image.new("RGB", (w, h), (12, 14, 20))
    
    # 1. Background checkerboard / floor
    # Load step 35 render (1080x1080)
    raw_im = Image.open("run_az75_step_35.png")
    
    # In raw_im, character center is x=752, y=574
    # Shift character to center (dx = 540 - 752 = -212)
    # Scale character slightly so height is ~880px (scale = 880 / 929 = 0.95)
    char_target_h = 880
    scale = char_target_h / 929.0
    new_w, new_h = int(raw_im.width * scale), int(raw_im.height * scale)
    scaled_im = raw_im.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Paste scaled_im onto canvas centered
    # We want character center at x = 540, y = 950 (middle of screen)
    cur_cx = int(752 * scale)
    cur_cy = int(574 * scale)
    paste_x = 540 - cur_cx
    paste_y = 950 - cur_cy
    
    # Fill background with ambient gradient
    for y in range(h):
        # subtle dark gradient from top to bottom
        ratio = y / h
        r = int(10 + 6 * ratio)
        g = int(12 + 6 * ratio)
        b = int(18 + 8 * ratio)
        canvas.paste((r, g, b), (0, y, w, y + 1))
        
    canvas.paste(scaled_im, (paste_x, paste_y))
    
    # 2. Text Boxes with EXACT Safe Margins
    # Top Margin = 200px (Completely safe from YouTube shelf crop of ~120px)
    # Bottom Margin = 200px (Completely safe from YouTube shelf & UI cutoff)
    
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    
    # --- Top Box ---
    # Width = 920 (Margin 80px on sides), Height = 170 (Y = 200 .. 370)
    top_box = [80, 200, w - 80, 370]
    ov_draw.rounded_rectangle(top_box, radius=20, fill=(0, 0, 0, 240), outline=(255, 120, 0, 220), width=4)
    
    # --- Bottom Box ---
    # Width = 920 (Margin 80px on sides), Height = 170 (Y = 1550 .. 1720)
    bot_box = [80, 1550, w - 80, 1720]
    ov_draw.rounded_rectangle(bot_box, radius=20, fill=(0, 0, 0, 240), outline=(255, 255, 255, 180), width=4)
    
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    
    # --- Typography ---
    f_text = get_font(92)
    
    # Top Text: '걸음마 시켰더니' (Orange #FF7A00 with crisp black stroke)
    draw_stroke_text(draw, (w // 2, 285), "걸음마 시켰더니", f_text, (255, 122, 0), stroke_color=(0, 0, 0), stroke_width=5)
    
    # Bottom Text: '마라토너 된 AI' (White #FFFFFF with crisp black stroke)
    draw_stroke_text(draw, (w // 2, 1635), "마라토너 된 AI", f_text, (255, 255, 255), stroke_color=(0, 0, 0), stroke_width=5)
    
    # Save the master thumbnail
    out_name = "thumbnail_shorts_perfect_safe.png"
    canvas.save(out_name, quality=98)
    print(f"Saved: {out_name}")
    
    # Also overwrite thumbnail_shorts_channel_style.png so user gets it immediately
    canvas.save("thumbnail_shorts_channel_style.png", quality=98)
    print("Overwritten: thumbnail_shorts_channel_style.png")
    
    # 3. Simulate YouTube Grid Preview (1:1.55 crop)
    # YouTube shelf crops to roughly 1080 x 1675
    crop_shelf_h = 1675
    crop_top = (h - crop_shelf_h) // 2
    shelf_preview = canvas.crop((0, crop_top, w, crop_top + crop_shelf_h))
    shelf_preview.save("preview_youtube_shelf_crop.png")
    print(f"Simulated Shelf Preview saved: preview_youtube_shelf_crop.png (Cropped top: {crop_top}px, bot: {crop_top}px)")

if __name__ == "__main__":
    build_perfect_thumbnail()

