"""Manual anatomical overlay, NOT monocular SMPL-X parameter estimation."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
src = Image.open(ROOT / '微信图片_20260918102730_136_455.jpg').convert('RGB')
src = src.resize((2048, 1218))
out = Image.new('RGB', (2848, 1378), '#101c29')
out.paste(src, (0, 0))
d = ImageDraw.Draw(out)
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
def font(n): return ImageFont.truetype(font_path, n)
yellow = '#ffe169'
# Anatomical estimates in the supplied mirrored UI, not tracker locations.
joints = {0:(1365,590), 1:(1325,600), 2:(1404,600),
          3:(1365,538), 6:(1365,455), 9:(1365,384),
          12:(1365,326),15:(1365,257),
          16:(1284,375),17:(1447,375),18:(1229,516),19:(1501,516),
          20:(1195,647),21:(1534,647),4:(1318,827),5:(1417,827),
          7:(1311,1086),8:(1422,1086),10:(1301,1126),11:(1433,1126)}
edges=[(0,1),(0,2),(0,3),(3,6),(6,9),(9,12),(12,15),
       (9,16),(9,17),(16,18),(17,19),(18,20),(19,21),
       (1,4),(2,5),(4,7),(5,8),(7,10),(8,11)]
for a,b in edges:
    d.line([joints[a],joints[b]], fill='#101c29', width=9)
    d.line([joints[a],joints[b]], fill=yellow, width=4)
for i,(x,y) in joints.items():
    d.ellipse((x-8,y-8,x+8,y+8),fill=yellow,outline='black',width=2)
    dx = -44 if i in (1,4,7,10,16,18,20) else 15
    if i in (0,3,6,9,12,15): dx=19
    label=str(i)
    box=d.textbbox((x+dx,y-16),label,font=font(23))
    d.rectangle((box[0]-3,box[1]-2,box[2]+3,box[3]+2),fill='#101c29')
    d.text((x+dx,y-16),label,font=font(23),fill=yellow)

lines=[('SMPL / SMPL-X 身体关节示意',32,yellow),
       ('手工解剖位置对齐 · 非参数预测结果',23,'#ffffff'),
       ('黄色：关节与骨段；绿色：原图 tracker',22,'#ffffff'),
       ('左右沿用截图标签（镜像显示已开启）',22,'#ffffff'),
       ('16 / 17  肩关节 → 上臂',29,yellow),
       ('18 / 19  肘关节 → 前臂',29,yellow),
       ('20 / 21  腕关节 → 手部',29,yellow),
       ('1 / 2      髋关节 → 大腿',29,yellow),
       ('4 / 5      膝关节 → 小腿',29,yellow),
       ('7 / 8      踝关节 → 脚部',29,yellow),
       ('10 / 11  前脚部（foot）',29,yellow),
       ('0 骨盆；3/6/9 脊柱；12 颈；15 头',24,'#ffffff'),
       ('如何比较传感器位置',30,yellow),
       ('左小臂 C151F / 右小臂 C8054',25,'#ffffff'),
       ('位于肘—腕之间：比较 18/19 的旋转。',23,'#ffffff'),
       ('左手 C1B50 / 右手 C305B',25,'#ffffff'),
       ('位于腕附近；实际绑在手还是前臂，',23,'#ffffff'),
       ('需以真实佩戴为准，不能只看距离。',23,'#ffffff'),
       ('小腿 tracker 靠近踝，但代表小腿骨段，',23,'#ffffff'),
       ('应比较 4/5 的旋转，而非 7/8。',23,'#ffffff'),
       ('腰部 C24C5：需确认安装朝向与偏置。',23,'#ffffff'),
       ('脚部白点未分配，不能充当脚部 IMU。',23,'#ffffff')]
y=50
for text,size,color in lines:
    d.text((2080,y),text,font=font(size),fill=color)
    y+=49 if size<29 else 63
d.text((45,1240),'注意：关节位置 ≠ 传感器位置。关节旋转描述骨段朝向；加速度取决于实际测量点。',font=font(30),fill=yellow)
d.text((45,1300),'仅绘制相关身体关节；肩带等中间关节省略。卡通比例下的标记为近似位置，不可用于定量标定。',font=font(26),fill='white')
dest=ROOT/'results/tracker_smpl_annotation'
dest.mkdir(parents=True,exist_ok=True)
out.save(dest/'tracker_smpl_body_overlay.png')
print(dest/'tracker_smpl_body_overlay.png')
