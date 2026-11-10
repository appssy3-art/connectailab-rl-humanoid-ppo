import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONT_BOLD = "C:/Windows/Fonts/malgunbd.ttf"
FONT_REG = "C:/Windows/Fonts/malgun.ttf"

def get_font(size):
    return ImageFont.truetype(FONT_BOLD, size)

def draw_stroke_text(draw, pos, text, font, fill_color, stroke_color=(0, 0, 0), stroke_width=6, anchor="mm"):
    x, y = pos
    draw.text((x, y), text, font=font, fill=fill_color, stroke_width=stroke_width, stroke_fill=stroke_color, anchor=anchor)

def create_big_thumbnails():
    print("🎨 [특대형 폰트 & 대형 캐릭터 썸네일 제작 시작]...")
    
    # Load hero images (1080x1080)
    im_gen0 = Image.open("hero_gen0_30.png")
    im_gen2500 = Image.open("hero_gen2500_30.png")
    
    # -------------------------------------------------------------
    # 1. 쇼츠 피드용 상하 2분할 특대형 썸네일 (thumbnail_shorts_big_split.png)
    # -------------------------------------------------------------
    w, h = 1080, 1920
    canvas1 = Image.new("RGB", (w, h), (10, 10, 14))
    
    # Top section for Gen 0 (Height 850)
    # Crop center of im_gen0 so character is centered
    crop_w0, crop_h0 = 1080, 850
    # im_gen0 is 1080x1080, character center is ~ (536, 571)
    top_box0 = (0, 150, 1080, 150 + crop_h0)
    gen0_crop = im_gen0.crop(top_box0)
    canvas1.paste(gen0_crop, (0, 0))
    
    # Bottom section for Gen 2500 (Height 850)
    # im_gen2500 character center is ~ (750, 565), shift slightly left
    left_shift = 150
    gen2500_shifted = Image.new("RGB", (1080, 1080), (15, 18, 25))
    gen2500_shifted.paste(im_gen2500, (-left_shift, 0))
    # crop height 850
    gen2500_crop = gen2500_shifted.crop((0, 100, 1080, 100 + crop_h0))
    canvas1.paste(gen2500_crop, (0, h - crop_h0))
    
    # Draw dark overlay boxes for text so text pops with 100% readability
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    
    # Top text banner box (y=40..250)
    ov_draw.rounded_rectangle([30, 40, w - 30, 240], radius=24, fill=(0, 0, 0, 225), outline=(255, 60, 60, 255), width=4)
    # Red badge for GEN 0
    ov_draw.rounded_rectangle([60, 65, 320, 135], radius=16, fill=(230, 35, 35, 255))
    
    # Center dividing bar (y=840..1070)
    ov_draw.rectangle([0, 840, w, 1070], fill=(12, 14, 20, 240), outline=(255, 200, 0, 255), width=4)
    
    # Bottom text banner box (y=1660..1880) - well above the bottom cutoff
    ov_draw.rounded_rectangle([30, 1660, w - 30, 1870], radius=24, fill=(0, 0, 0, 230), outline=(0, 230, 255, 255), width=4)
    # Cyan badge for GEN 2500
    ov_draw.rounded_rectangle([60, 1685, 380, 1755], radius=16, fill=(0, 180, 240, 255))
    
    canvas1 = Image.alpha_composite(canvas1.convert("RGBA"), overlay).convert("RGB")
    draw1 = ImageDraw.Draw(canvas1)
    
    # Text for Top
    f_badge = get_font(42)
    f_big = get_font(78)
    f_huge = get_font(94)
    
    # Top box texts
    draw1.text((190, 100), "0세대 꽈당", font=f_badge, fill=(255, 255, 255), anchor="mm")
    draw_stroke_text(draw1, (w // 2 + 100, 100), "태어나자마자 안면 슬라이딩", get_font(46), (255, 220, 220), stroke_width=3)
    draw_stroke_text(draw1, (w // 2, 185), "걸음마 시켰더니 바닥 청소 중", f_big, (255, 80, 80), stroke_width=6)
    
    # Center dividing bar text
    draw_stroke_text(draw1, (w // 2, 905), "AI에게 걷는 법을 안 가르쳐줌", get_font(52), (255, 225, 0), stroke_width=4)
    draw_stroke_text(draw1, (w // 2, 995), "250만 번 구르고 생긴 일!", f_huge, (255, 255, 255), stroke_width=8)
    
    # Bottom box texts
    draw1.text((220, 1720), "250만 세대", font=f_badge, fill=(0, 0, 0), anchor="mm")
    draw_stroke_text(draw1, (w // 2 + 130, 1720), "시속 5.6km 마라톤 질주 완성", get_font(46), (220, 255, 255), stroke_width=3)
    draw_stroke_text(draw1, (w // 2, 1810), "인간 초월! 마라토너 탄생", f_big, (0, 240, 255), stroke_width=6)
    
    canvas1.save("thumbnail_shorts_big_split.png", quality=98)
    print("Saved: thumbnail_shorts_big_split.png")
    
    # -------------------------------------------------------------
    # 2. 채널 맞춤형 스타일 (thumbnail_shorts_channel_style.png)
    #    유저 채널의 [걸음마 포기하고 롤러를 선택한 AI] 와 100% 동일한 블랙박스 대형 폰트 구도
    # -------------------------------------------------------------
    canvas2 = Image.new("RGB", (w, h), (10, 10, 14))
    
    # Put zoomed Gen 2500 character right in the center (taking 70% of screen)
    # Resize im_gen2500 to fill height ~1200
    char_h = 1350
    scale2 = char_h / im_gen2500.height
    char_w = int(im_gen2500.width * scale2)
    im2_scaled = im_gen2500.resize((char_w, char_h), Image.Resampling.LANCZOS)
    
    # Paste centered
    canvas2.paste(im2_scaled, ((w - char_w) // 2 - 120, 280))
    
    overlay2 = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ov2_draw = ImageDraw.Draw(overlay2)
    
    # Top Black Box (exact style as "걸음마 포기하고")
    box_top_y = 100
    ov2_draw.rectangle([50, box_top_y, w - 50, box_top_y + 160], fill=(0, 0, 0, 245), outline=(255, 120, 0, 255), width=5)
    
    # Floating Mini Fail Stamp at middle-left
    ov2_draw.rounded_rectangle([60, 520, 420, 630], radius=18, fill=(220, 30, 30, 240), outline=(255, 255, 255, 255), width=4)
    
    # Bottom Black Box (exact style as "롤러를 선택한 AI")
    box_bot_y = 1520
    ov2_draw.rectangle([50, box_bot_y, w - 50, box_bot_y + 280], fill=(0, 0, 0, 245), outline=(0, 240, 255, 255), width=5)
    
    canvas2 = Image.alpha_composite(canvas2.convert("RGBA"), overlay2).convert("RGB")
    draw2 = ImageDraw.Draw(canvas2)
    
    # Top Text
    f_chan_title = get_font(98)
    draw_stroke_text(draw2, (w // 2, box_top_y + 80), "걸음마 시켰더니", f_chan_title, (255, 140, 30), stroke_width=6)
    
    # Mini Stamp Text
    draw_stroke_text(draw2, (240, 575), "0세대: 0초 컷 꽈당", get_font(38), (255, 255, 255), stroke_width=3)
    
    # Bottom Text (2 lines, huge)
    draw_stroke_text(draw2, (w // 2, box_bot_y + 75), "250만 번 구르고", get_font(78), (255, 255, 255), stroke_width=5)
    draw_stroke_text(draw2, (w // 2, box_bot_y + 190), "마라토너 된 AI", get_font(102), (0, 240, 255), stroke_width=8)
    
    canvas2.save("thumbnail_shorts_channel_style.png", quality=98)
    print("Saved: thumbnail_shorts_channel_style.png")
    
    # -------------------------------------------------------------
    # 3. 유튜브 가로 16:9 대형 썸네일 (thumbnail_youtube_16x9_big.png)
    # -------------------------------------------------------------
    w3, h3 = 1920, 1080
    canvas3 = Image.new("RGB", (w3, h3), (10, 12, 18))
    
    # Half width = 960 each
    crop_w3 = 960
    # Left: Gen 0, Right: Gen 2500
    scale3 = h3 / im_gen0.height
    im0_res = im_gen0.resize((int(im_gen0.width * scale3), h3), Image.Resampling.LANCZOS)
    im2500_res = im_gen2500.resize((int(im_gen2500.width * scale3), h3), Image.Resampling.LANCZOS)
    
    canvas3.paste(im0_res.crop((60, 0, 60 + crop_w3, h3)), (0, 0))
    canvas3.paste(im2500_res.crop((0, 0, crop_w3, h3)), (crop_w3, 0))
    
    # Dark gradients at top and bottom
    overlay3 = Image.new("RGBA", (w3, h3), (0, 0, 0, 0))
    ov3_draw = ImageDraw.Draw(overlay3)
    
    # Center dividing lightning line
    ov3_draw.line([(crop_w3 - 4, 0), (crop_w3 - 4, h3)], fill=(255, 200, 0, 255), width=8)
    ov3_draw.line([(crop_w3 + 4, 0), (crop_w3 + 4, h3)], fill=(255, 255, 255, 255), width=3)
    
    # Top banner bar
    ov3_draw.rectangle([60, 40, w3 - 60, 190], fill=(0, 0, 0, 230), outline=(255, 210, 0, 255), width=5)
    
    # Bottom boxes
    ov3_draw.rectangle([60, h3 - 220, crop_w3 - 50, h3 - 50], fill=(0, 0, 0, 230), outline=(255, 60, 60, 255), width=4)
    ov3_draw.rectangle([crop_w3 + 50, h3 - 220, w3 - 60, h3 - 50], fill=(0, 0, 0, 230), outline=(0, 230, 255, 255), width=4)
    
    canvas3 = Image.alpha_composite(canvas3.convert("RGBA"), overlay3).convert("RGB")
    draw3 = ImageDraw.Draw(canvas3)
    
    # Top text
    draw_stroke_text(draw3, (w3 // 2, 115), "AI에게 걷기를 시켰더니 250만 번 구른 결과", get_font(68), (255, 255, 255), stroke_width=6)
    
    # Left text
    draw_stroke_text(draw3, (crop_w3 // 2, h3 - 165), "0세대: 0초 컷 대참사", get_font(44), (255, 180, 180), stroke_width=3)
    draw_stroke_text(draw3, (crop_w3 // 2, h3 - 100), "바닥 청소 슬라이딩 ㅋㅋㅋ", get_font(52), (255, 70, 70), stroke_width=6)
    
    # Right text
    draw_stroke_text(draw3, (w3 - crop_w3 // 2, h3 - 165), "250만 세대: 마라토너 완성", get_font(44), (180, 240, 255), stroke_width=3)
    draw_stroke_text(draw3, (w3 - crop_w3 // 2, h3 - 100), "시속 5.6km 인간 초월 질주!", get_font(52), (0, 240, 255), stroke_width=6)
    
    canvas3.save("thumbnail_youtube_16x9_big.png", quality=98)
    print("Saved: thumbnail_youtube_16x9_big.png")

if __name__ == "__main__":
    create_big_thumbnails()
