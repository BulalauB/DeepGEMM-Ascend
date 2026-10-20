"""Render source-derived, non-scaled MegaMoE pipeline diagrams (Pillow)."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).absolute().parents[1] / 'figures' / 'megamoe_execution'
OUT.mkdir(parents=True, exist_ok=True)
FONT = 'C:/Windows/Fonts/msyh.ttc'
COLORS = {'s':'#f6c6c6', 'm2':'#add8f5', 'v':'#ffe89b', 'm3':'#bce9c5',
          'm1':'#c9d4f4', 'cube':'#d8bff0', 'fix':'#f4cf9f', 'wait':'#e9edf0'}

class Chart:
    def __init__(self, title, subtitle, lanes, notes):
        self.lanes = lanes
        self.w = 1640
        self.base = 160
        self.dy = 105
        self.h = self.base + len(lanes)*self.dy + 85 + len(notes)*35
        self.im = Image.new('RGB', (self.w, self.h), '#ffffff')
        self.d = ImageDraw.Draw(self.im)
        self.text((38,25), title, 32)
        self.text((38,78), subtitle, 21, '#52606b')
        self.left, self.right = 245, 1590
        for i, label in enumerate(lanes):
            y = self.base+i*self.dy
            self.text((35,y+17),label,23)
            self.d.line((self.left,y+35,self.right,y+35),fill='#b9c2ca',width=2)
        y = self.base+len(lanes)*self.dy
        self.d.line((self.left,y,self.right,y), fill='#65717c',width=2)
        self.d.polygon([(self.right,y),(self.right-12,y-6),(self.right-12,y+6)], fill='#65717c')
        self.text((self.left,y+10),'逻辑推进方向 →   方块宽度、空白和对齐不代表实测周期',20,'#52606b')
        for j,note in enumerate(notes): self.text((38,y+57+j*35),note,21,'#39434d')

    def text(self, xy, value, size=22, fill='#202b35'):
        self.d.text(xy,value,font=ImageFont.truetype(FONT,size),fill=fill)

    def box(self,lane,start,end,label,kind,tag=None):
        x0=self.left+(self.right-self.left)*start/100
        x1=self.left+(self.right-self.left)*end/100
        y=self.base+lane*self.dy
        self.d.rectangle((x0,y,x1,y+70),fill=COLORS[kind],outline='#56616c',width=2)
        lines=label.split('\n')
        for j,line in enumerate(lines):
            font=ImageFont.truetype(FONT,21)
            width=self.d.textbbox((0,0),line,font=font)[2]
            assert width <= x1-x0-2, (label,width,x1-x0)
            self.d.text(((x0+x1-width)/2,y+9+j*28),line,font=font,fill='#202b35')
        if tag: self.text((x0,y-27),tag,18,'#52606b')
        return (x0,x1,y)

    def arrow(self,from_lane,from_t,to_lane,to_t):
        x0=self.left+(self.right-self.left)*from_t/100
        x1=self.left+(self.right-self.left)*to_t/100
        down=to_lane>from_lane
        y0=self.base+from_lane*self.dy+(72 if down else -2)
        y1=self.base+to_lane*self.dy+(-2 if down else 72)
        self.d.line([(x0,y0),(x1,y1)],fill='#667583',width=2)
        import math
        a=math.atan2(y1-y0,x1-x0)
        pts=[(x1,y1),(x1-10*math.cos(a-.5),y1-10*math.sin(a-.5)),(x1-10*math.cos(a+.5),y1-10*math.sin(a+.5))]
        self.d.polygon(pts,fill='#667583')

    def save(self,name):
        self.im.save(OUT / (name+'.png'))

# A lane is always an execution resource of ONE explicitly named AIV.
# These are feasible dependency schedules, not a simulated or measured trace.
c=Chart('AIV0：Scalar / Vector / MTE2 / MTE3',
        '稳态权重子块 i、i+1、i+2；元数据已完成；只画 AIV0，Cube 内部搬运省略',
        ['Scalar','Vector','MTE2','MTE3'],
        ['① Scalar：解析 task / K / N，计算地址和槽号，发射三条流水的指令。',
         '② MTE2：GM packed FP4 → UB；③ Vector：展开 FP8 并排成 NZ；④ MTE3：NZ UB → AIC L1B。',
         '②→③：FP4Full；③→④：NZFull；④还要等 AIC 归还 L1BEmpty，完成后通知 AIC L1BFull。',
         'FP4 槽3份、NZ槽2份、L1B槽4份；图中重叠要求对应 empty credit 已满足。',
         'Cube 在 L1BFull 且激活/scale 就绪后消费；此处不把 Cube 内部搬运混入 AIV0 的 MTE2/MTE3。'])
c.box(0,0,20,'① i：地址 / 发射','s')
c.box(0,22,42,'① i+1：地址 / 发射','s')
c.box(0,44,64,'① i+2：地址 / 发射','s')
for t,j in [(0,'i'),(20,'i+1'),(40,'i+2')]:
    c.box(2,5+t,23+t,f'② {j}\nGM → FP4 UB','m2')
    c.box(1,23+t,43+t,f'③ {j}\nFP4 → FP8 NZ','v')
    c.box(3,43+t,60+t,f'④ {j}\nNZ UB → L1B','m3')
    c.arrow(2,23+t,1,23+t)
c.save('05_aiv0_resources')

c=Chart('AIV1：epilogue、pull、push 共用这一组四条流水',
        '一个可行的稳态片段：G1 task 的前128行（valid M=256）；旧块 p 已 G2 就绪，未来块 f 可 pull',
        ['Scalar','Vector','MTE2','MTE3'],
        ['E1~E4：同一个 M128 槽的四次32行 SwiGLU / 乘权 / 量化；E1 前要求 Cube 的 FIX Full。',
         'P0/P1：旧块 p 的两次 push 推进；D0/D1：未来块 f 的两次 pull 推进；每次最多两行。',
         '图中每次通信仅展开一行；若本次有第二行，也排入同一 MTE2/MTE3 队列。',
         'MTE2 上的路由权重、push读取、pull读取不能同时执行；MTE3 上三种写回也按队列顺序执行。',
         'E4 后才提交本128行槽的 FP8/SF 写回；写完归还 Cube 的 FIX credit；此时整个 M256 task 尚未结束。',
         'E、P、D 对应不同数据，不是同一块的 G1→G2→通信；这里不宣称固定每轮都能成功 push/pull。'])
for start,end,label in [(0,10,'E1'),(10,20,'P0'),(20,30,'E2'),(30,40,'D0'),
                         (40,50,'E3'),(50,60,'P1'),(60,70,'E4'),(70,80,'D1'),(80,100,'后续推进')]:
    c.box(0,start,end,label,'s')
for start,end,label in [(8,25,'E1 · 32行'),(28,45,'E2 · 32行'),(48,65,'E3 · 32行'),(68,85,'E4 · 32行')]:
    c.box(1,start,end,label,'v')
c.box(2,0,8,'权重','m2')
c.box(2,12,26,'P0\n本地GM读','m2')
c.box(2,32,46,'D0\npeer GM读','m2')
c.box(2,52,66,'P1\n本地GM读','m2')
c.box(2,72,86,'D1\npeer GM读','m2')
c.box(3,26,40,'P0\n回源GM写','m3')
c.box(3,46,60,'D0\n本地环写','m3')
c.box(3,66,80,'P1\n回源GM写','m3')
c.box(3,85,93,'E 写回','m3')
c.box(3,93,100,'D1 写','m3')
c.save('06_aiv1_resources')

print(OUT)

# Unified view: both AIVs and Cube share the same logical x axis.
class Combined:
    def __init__(self):
        self.im = Image.new('RGB', (1900, 1560), '#ffffff')
        self.d = ImageDraw.Draw(self.im)
        self.left, self.right = 260, 1850
        self.ys = [200,300,400,500,680,860,960,1060,1160]
        self.txt(35,25,'MegaMoE：AIV0 / Cube / AIV1 统一流水图',34)
        self.txt(35,82,'同一横轴 · q 为当前 G1 task，r 为下一 G1 task；P 回传旧块，D 预拉后续块',23)
        self.txt(35,125,'画出依赖允许的一种重叠；非实测周期。Cube 内部搬运省略，输出以 FIX 完成边界连接。',22)
        for y,label in [(166,'AIV0'),(646,'Cube'),(826,'AIV1')]:
            self.txt(35,y,label,26)
        for i,label in enumerate(['Scalar','Vector','MTE2','MTE3','MAD','Scalar','Vector','MTE2','MTE3']):
            self.txt(100,self.ys[i]+20,label,23)
            self.d.line((self.left,self.ys[i]+35,self.right,self.ys[i]+35),fill='#c4ccd2',width=2)
        for y in [605,785]: self.d.line((35,y,1850,y),fill='#9ca9b3',width=2)

    def txt(self,x,y,s,size=21,color='#202b35'):
        self.d.text((x,y),s,font=ImageFont.truetype(FONT,size),fill=color)

    def x(self,t): return self.left+(self.right-self.left)*t/140

    def box(self,lane,a,b,s,kind):
        x0,x1,y=self.x(a),self.x(b),self.ys[lane]
        self.d.rectangle((x0,y,x1,y+70),fill=COLORS[kind],outline='#56616c',width=2)
        for j,line in enumerate(s.split('\n')):
            font=ImageFont.truetype(FONT,20)
            w=self.d.textbbox((0,0),line,font=font)[2]
            assert w<=x1-x0-4,(line,w,x1-x0)
            self.d.text(((x0+x1-w)/2,y+10+j*27),line,font=font,fill='#202b35')

    def link(self,points):
        import math
        self.d.line(points,fill='#52697d',width=3)
        (x0,y0),(x1,y1)=points[-2:]
        a=math.atan2(y1-y0,x1-x0)
        self.d.polygon([(x1,y1),(x1-12*math.cos(a-.45),y1-12*math.sin(a-.45)),
                       (x1-12*math.cos(a+.45),y1-12*math.sin(a+.45))],fill='#52697d')

u=Combined()
u.box(0,0,10,'q：发射','s')
u.box(0,10,22,'q：下一 K','s')
u.box(0,34,50,'r：地址/发射','s')
u.box(2,0,12,'q K−2\nFP4读取','m2')
u.box(1,12,22,'q K−2\n展开 NZ','v')
u.box(3,22,30,'q K−2\n写 L1B','m3')
u.box(2,12,24,'q K−1\nFP4读取','m2')
u.box(1,24,34,'q K−1\n展开 NZ','v')
u.box(3,34,42,'q K−1\n写 L1B','m3')
u.box(2,36,48,'r K0\nFP4读取','m2')
u.box(1,48,58,'r K0\n展开 NZ','v')
u.box(3,58,66,'r K0\n写 L1B','m3')
u.box(4,30,42,'q 倒数K块\nMAD','cube')
u.box(4,42,54,'q 最后K块\nMAD','cube')
u.box(4,66,104,'r：后续 G1 MAD …\nFIX 输出仍等 AIV1 归还结果槽','cube')
u.txt(u.x(68),583,'r 后续 K 子块仍按相同链持续供给（未逐一展开）',20)
u.link([(u.x(30),572),(u.x(30),678)])
u.txt(u.x(31),612,'L1BFull + A/SF/L0 ready',19)
u.link([(u.x(66),572),(u.x(66),678)])
u.link([(u.x(54),752),(u.x(52),775),(u.x(52),944),(u.x(65),944),(u.x(65),958)])
u.txt(u.x(1),796,'q：最后 K → FIX 完成 → EpilogueFull',20)
for a,b,s in [(54,68,'E1'),(68,78,'P0'),(78,88,'E2'),(88,98,'D0'),
              (98,108,'E3'),(108,118,'P1'),(118,128,'E4'),(128,140,'D1')]:
    u.box(5,a,b,s,'s')
for a,b,s in [(65,80,'E1 · 32行'),(85,100,'E2 · 32行'),(105,120,'E3 · 32行'),(125,140,'E4 · 32行')]:
    u.box(6,a,b,s,'v')
u.box(7,56,65,'路由权重','m2')
u.box(7,73,87,'P0\n本地读','m2')
u.box(7,93,107,'D0\npeer 读','m2')
u.box(7,113,127,'P1\n本地读','m2')
u.box(7,132,140,'D1读…','m2')
u.box(8,87,101,'P0\n回源写','m3')
u.box(8,107,121,'D0\n本地环写','m3')
u.box(8,127,140,'P1 写…','m3')
u.txt(260,1255,'→ 后续：E4 完成后，MTE3 排队写本128行槽的 FP8/SF；完成后归还 Cube 的 FIX credit。',22)
u.txt(35,1310,'范围：q 取 M256/N256 配置；上方只展开 q 最后两个 K-block，以及 r 的第一个 K-block。',22)
u.txt(35,1350,'前提：q/r 的输入已 Dispatched；r 是调度中的下一 G1；P 的整块 G2 输出已 ready；D 的输入环可用。',22)
u.txt(35,1390,'所有权：两组 Scalar/MTE2/MTE3 各自独立；同一 AIV1 内 E/P/D 共享队列，且 E/P/D 来自不同数据。',22)
u.txt(35,1430,'同步：每份权重仍按 MTE2 → Vector → MTE3 → Cube；AIV1 的 Vector 另等路由权重 MTE2 Full。',22)
u.txt(35,1470,'横轴只表示可行先后与重叠，不表示实际速度；省略区间和 Cube 内部搬运不能据图估算空闲率。',22)
u.im.save(str(OUT / '07_combined_pipeline.png'))
