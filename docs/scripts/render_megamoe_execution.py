"""Source-derived MegaMoE figures: overall dependencies and local resource schedules.

Run from any directory. Requires Pillow. Override font with --font or MEGAMOE_FONT.
All widths are illustrative; no duration estimates or profiler data are used.
"""
from pathlib import Path
import argparse
import math
import os
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--font', default=os.environ.get('MEGAMOE_FONT', 'C:/Windows/Fonts/msyh.ttc'))
parser.add_argument('--out', default=str(Path(__file__).absolute().parents[1] / 'figures' / 'megamoe_execution'))
args = parser.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
FONT = args.font
COLORS = {'s':'#f6c6c6', 'm2':'#add8f5', 'v':'#ffe89b', 'm3':'#bce9c5',
          'cube':'#d8bff0', 'neutral':'#e9edf0'}
GENERATED = []

def label(draw, x, y, text, size=22, color='#202b35'):
    draw.text((x,y), text, font=ImageFont.truetype(FONT,size), fill=color)

def rect_text(draw, bounds, text, kind, size=22):
    x0,y0,x1,y1 = bounds
    draw.rectangle(bounds,fill=COLORS[kind],outline='#596772',width=2)
    lines = text.split('\n')
    font = ImageFont.truetype(FONT,size)
    assert 29*len(lines) <= y1-y0-4, text
    for j,line in enumerate(lines):
        w=draw.textbbox((0,0),line,font=font)[2]
        assert w <= x1-x0-8, (line,w,x1-x0)
        draw.text(((x0+x1-w)/2,y0+9+29*j),line,font=font,fill='#202b35')

def arrow(draw, points):
    draw.line(points, fill='#52697d', width=2)
    (x0,y0),(x1,y1)=points[-2:]
    a=math.atan2(y1-y0,x1-x0)
    draw.polygon([(x1,y1),(x1-11*math.cos(a-.45),y1-11*math.sin(a-.45)),
                  (x1-11*math.cos(a+.45),y1-11*math.sin(a+.45))],fill='#52697d')

class Chart:
    def __init__(self,title,subtitle,notes):
        self.base,self.dy,self.left,self.right=160,108,205,1750
        self.height=self.base+4*self.dy+90+len(notes)*36
        self.im=Image.new('RGB',(1800,self.height),'white')
        self.d=ImageDraw.Draw(self.im)
        label(self.d,35,25,title,32)
        label(self.d,35,83,subtitle,22)
        for i,s in enumerate(['Scalar','Vector','MTE2','MTE3']):
            y=self.base+i*self.dy
            label(self.d,35,y+20,s,25)
            self.d.line((self.left,y+36,self.right,y+36),fill='#bcc7cf',width=2)
        y=self.base+4*self.dy
        arrow(self.d,[(self.left,y),(self.right,y)])
        label(self.d,self.left,y+12,'局部逻辑推进 →   仅表示一种可行依赖排布；宽度、空白不表示周期',21)
        for j,s in enumerate(notes): label(self.d,35,y+65+36*j,s,22)

    def box(self,lane,a,b,text,kind=None):
        if kind is None: kind=['s','v','m2','m3'][lane]
        x0=self.left+(self.right-self.left)*a/100
        x1=self.left+(self.right-self.left)*b/100
        y=self.base+lane*self.dy
        rect_text(self.d,(x0,y,x1,y+76),text,kind,21)

    def empty(self,lane,text):
        label(self.d,self.left+20,self.base+lane*self.dy+1,text,21,'#52616e')

    def save(self,name):
        self.im.save(str(OUT/name))
        GENERATED.append(name)

# Layer 2: the axis is a stage dependency chain, not a hardware timeline.
im=Image.new('RGB',(1900,940),'white')
d=ImageDraw.Draw(im)
label(d,35,25,'MegaMoE 整体协作：阶段依赖与两条旁路',34)
label(d,35,85,'这张图回答“谁生产什么、谁等谁”；横向位置不代表执行耗时，不展开各自的硬件队列。',23)
rect_text(d,(40,150,490,245),'调用前：FP8 token / scale / top-k 已给定','neutral')
rect_text(d,(700,150,1160,245),'AIV0：权重持续供给\nFP4 → FP8 NZ → AIC L1B','v')
nodes=[('前处理\nAIV0 + 入口全部AIV','s'),('Dispatch pull\nAIV1','m2'),('GMM1\nAIC / Cube','cube'),
       ('激活 / 量化\nAIV1','v'),('GMM2\nAIC / Cube','cube'),('Combine push\nAIV1','m3'),('最终归并\nAIV0 + AIV1','v')]
for i,(s,k) in enumerate(nodes):
    x=40+260*i
    rect_text(d,(x,355,x+230,455),s,k)
    if i<6: arrow(d,[(x+230,405),(x+260,405)])
for x,s in [(40,'metadata ready'),(300,'Dispatched = 32'),(560,'对应槽 FIX Full'),(820,'G1 全列写完'),(1080,'G2 全列写完'),(1340,'对应结果接收可读')]:
    label(d,x,490,s,21)
arrow(d,[(265,245),(265,300),(155,300),(155,353)])
arrow(d,[(220,353),(220,315),(630,315),(630,197),(698,197)])
label(d,330,273,'元数据就绪后供给 routed 权重',20)
arrow(d,[(930,245),(930,295),(675,295),(675,353)])
arrow(d,[(930,295),(1195,295),(1195,353)])
rect_text(d,(40,620,680,720),'可选 shared：AIC + AIV1\n本地 G1 → 激活量化 → G2 → 本地 shared 接收槽','neutral')
arrow(d,[(680,670),(1715,670),(1715,457)])
label(d,40,755,'shared 可在 routed 元数据期间先计算；shared 权重已为 FP8 NZ，不走 routed FP4 展开链。',23)
label(d,40,800,'同一块遵守主链依赖；不同块的通信、权重准备、GEMM 和 epilogue 可以重叠。',23)
label(d,40,845,'任务按 G1 lookahead 调度，不保证 G1/G2 一比一交替；最终归并读取有效 top-k 槽及可选 shared 槽。',23)
im.save(str(OUT/'00_overview.png'))
GENERATED.append('00_overview.png')

c=Chart('前处理 · AIV0：SIMT 阶段、汇合与发布',
        '粒度：参与元数据的一个 AIV0 worker；M0 包含所有 rank 的入口同步，不是单核独立清理',
        ['M1~M5 在一次 SIMT VF 内执行；G0/G1/G2 分别对应三次 grid_sync，含必要的 fence/worker 汇合。',
         'M1计数 → G0 → M2前缀 → M3压紧索引 → G1 → M4发送64-bit元数据 → G2 → M5发布count/arrival。',
         'Scalar 等 V→S MetadataDone 后，M6 等各专家 P 个源到齐、32个 AIV0 汇合块数，发布 metadata_ready。',
         '图中两行 MTE 留空：该 VF 内是 SIMT 指针读写/原子操作，源码没有显式 token DMA 链。',
         'worker 内每阶段还有线程同步；此图的 G0/G1/G2 是本 rank worker 同步，不是三次跨 rank barrier。'])
c.box(0,0,12,'M0：清理\n调用入口同步')
for a,b,s in [(14,27,'M1：计数'),(27,31,'G0'),(31,43,'M2：前缀'),(43,56,'M3：压紧'),
              (56,60,'G1'),(60,76,'M4：发送'),(76,80,'G2'),(80,89,'M5\n发布计数')]: c.box(1,a,b,s)
c.box(0,87,93,'等 VF')
c.box(0,93,100,'M6\n证书')
c.empty(2,'无显式 MTE2 搬运；SIMT GM 访问不在这里强行归类')
c.empty(3,'无显式 MTE3 搬运；count/arrival 由 VF 内的 GM 写、fence 和原子操作发布')
c.save('01_metadata.png')

c=Chart('Dispatch pull · AIV1：两行搬运与块尾确认',
        '粒度：本核该块最后两条有效路由行 r0/r1；已满足 metadata 和输入环空间条件',
        ['D1：解 metadata、还原 source/token/top-k、计算 peer 输入和本地环地址；r0/r1 行号相隔32。',
         'D2：FP8 token 与 scale 两次 MTE2 读取；D3：两次 MTE3 写本地输入环；D4：Scalar保存权重/身份。',
         'D5：末批发出后登记 pending；下一次推进等 PullDone、DSB，再贡献一次 Dispatched，32核到齐才放行 G1。',
         '两份 UB 槽；每次推进最多两行。槽回收是 MTE3→MTE2 PullEmpty，不是等整块 G1 做完。',
         '本图省略块入口的专家/source区间准备；元数据已给出权重与索引，输入已是FP8，无Vector量化。'])
for a,b,s in [(0,16,'D1 r0\n身份 / 地址'),(20,32,'D4 r0\n保存身份'),(34,50,'D1 r1\n身份 / 地址'),
              (54,66,'D4 r1\n保存身份'),(82,93,'D5\n等写完'),(93,100,'贡献\n计数')]: c.box(0,a,b,s)
c.box(2,16,34,'D2 r0\npeer GM → UB0')
c.box(2,50,68,'D2 r1\npeer GM → UB1')
c.box(3,34,52,'D3 r0\nUB0 → 本地环')
c.box(3,68,86,'D3 r1\nUB1 → 本地环')
c.empty(1,'此链不做 Vector 算术；Vector 可服务其他推进步的 epilogue')
c.save('02_dispatch.png')

c=Chart('权重准备 · AIV0：不同子块的三级数据流水',
        '粒度：相邻权重子块 i / i+1 / i+2；只展开 AIV0，AIC 内部搬运另见文档',
        ['W1：Scalar 解析 task/K/N、计算地址和槽号，发射后续指令。',
         'W2：GM packed FP4 → UB；W3：Vector 展开 FP8 NZ；W4：NZ UB → AIC L1B。',
         'W2→W3 等 FP4Full；W3→W4 等 NZFull；W4 另等 AIC 的 L1BEmpty，完成通知 L1BFull。',
         'FP4 UB为3槽，NZ UB为2槽，L1B为4槽；分别在V、MTE3、AIC MTE1读完后回收。'])
for a,b,s in [(0,20,'W1 i'),(22,42,'W1 i+1'),(44,64,'W1 i+2')]: c.box(0,a,b,s)
for t,j in [(0,'i'),(20,'i+1'),(40,'i+2')]:
    c.box(2,5+t,23+t,'W2 '+j+'\nGM → FP4 UB')
    c.box(1,23+t,43+t,'W3 '+j+'\n展开 / NZ')
    c.box(3,43+t,60+t,'W4 '+j+'\nNZ UB → L1B')
c.save('03_weights.png')

c=Chart('G1 epilogue · AIV1：四次32行计算，一次整槽写回',
        '粒度：一个有效 M128 的完整 G1 task；若 task 为 M256，还需处理第二个槽后才能发布该 task 完成',
        ['E0：Scalar探测 issued；issued只证明已提交，Vector还须等待对应FIX Full及MTE2权重Full。',
         'E1：MTE2读本槽路由权重；E2a~d：每次最多32行，SwiGLU→乘权→amax/scale→FP8。',
         'E3：整个128行槽量化完才提交FP8和scale写GM；MTE3完成边界归还对应FIX Empty。',
         'E4：完整task结束，Scalar等stores、DSB后贡献一次Linear1Ready；全块 I/128 个task到齐才允许G2。',
         'shared不读/乘路由权重；G2 epilogue只等FIX→GM完成并发布Linear2Ready，没有再次激活量化。'])
for a,b,s in [(0,15,'E0：探测\n地址 / 发射'),(18,32,'下一32行'),(35,49,'下一32行'),(52,67,'最后32行'),
              (85,95,'E4\n等stores'),(95,100,'贡献')]: c.box(0,a,b,s)
c.box(2,3,15,'E1\n路由权重')
for a,b,s in [(15,30,'E2a · 32行'),(32,47,'E2b · 32行'),(49,64,'E2c · 32行'),(66,81,'E2d · 32行')]: c.box(1,a,b,s)
c.box(3,81,95,'E3：FP8 + SF\n写 GM / 归还槽')
c.save('04_epilogue.png')

c=Chart('Combine push · AIV1：逐行回源，批量确认发送完成',
        '粒度：一次待确认批次的最后两行；本块所有 G2 列已就绪，输出环和返回身份可读',
        ['P1：Scalar读[source rank, combine row]、计算源/目的地址；P2：MTE2读本地整行H维BF16。',
         'P3：MTE3写源rank独立top-k槽；P4：到确认边界才等stores、DSB并贡献CombineSent。',
         '每次最多两行，四个push UB槽；通常每4个M-block确认一次，尾部按分支强制确认。',
         '4个UB槽与4块确认间隔是不同概念；计数每块等32个AIV1，不按行数累加。',
         '回源不做Vector乘权或数值AtomicAdd；不同专家先写独立槽，最终reduce才求和。'])
for a,b,s in [(0,16,'P1 r0\n身份 / 地址'),(34,50,'P1 r1\n身份 / 地址'),(82,94,'P4\n等stores'),(94,100,'贡献')]: c.box(0,a,b,s)
c.box(2,16,34,'P2 r0\n本地 GM → UB')
c.box(2,50,68,'P2 r1\n本地 GM → UB')
c.box(3,34,52,'P3 r0\nUB → 源rank')
c.box(3,68,86,'P3 r1\nUB → 源rank')
c.empty(1,'无 Vector 数值处理；CombineSent 证明发送完成，可回收旧输出环')
c.save('05_combine.png')

c=Chart('最终归并 · 每个 AIV：按通道 chunk 的双缓冲',
        '粒度：本AIV最后一个token的两个尾部通道chunk c0/c1；AIV0、AIV1各自执行，接收可读条件已满足',
        ['R0：判断路由有效性与接收可读，计算chunk地址；overlap检查真实生产块和peer完成前缀。',
         'R1：MTE2依次读该chunk所有有效top-k槽及可选shared槽；Full覆盖这些读取的完成。',
         'R2：BF16转FP32累加，再转BF16；R3：MTE3写最终output；MTE3读完后归还该槽Empty。',
         '只有两份reduce槽；这里不再乘路由权重。尾部放大chunk前先排空旧reduce槽，避免新旧UB视图重叠。',
         '本图不能替代接收可读证明；当前工作区非overlap屏障偏差见文档，不能据图假定数据已到齐。'])
c.box(0,0,15,'R0\n可读 / 地址')
c.box(0,18,31,'c1 地址\n发射')
c.box(0,83,100,'发射尾部握手\nAIC 等写回完成')
c.box(2,15,32,'R1 c0\n多路 GM → UB0')
c.box(2,32,49,'R1 c1\n多路 GM → UB1')
c.box(1,32,49,'R2 c0\nFP32 求和')
c.box(1,49,66,'R2 c1\nFP32 求和')
c.box(3,49,66,'R3 c0\nUB0 → output')
c.box(3,66,83,'R3 c1\nUB1 → output')
c.save('06_reduce.png')

c=Chart('AIV1 资源争用：E / P / D 共用四条队列',
        '附图：可行稳态片段；E为当前M256 task前128行槽，P为已就绪旧块，D为可预拉后续块',
        ['Scalar先推进epilogue，再尝试push/pull；初始优先pull，稳态偏好随成功推进而切换，本图从偏好push处截取。',
         'Ea~Ed分别32行；通信每次最多两行，本图只展开一次推进中的一行，不能由条形数推算调用数。',
         'MTEb：权重读 / P读 / D读依次排队；MTEc：P写 / D写 / 激活写依次排队，同一队列不画重叠。',
         'Ed后先提交激活写，再提交Db写；此处是M256前半槽，不立即贡献完整task的Linear1Ready。',
         '它解释跨块重叠与资源竞争，不代替同块 Dispatch→G1→激活→G2→Combine 的先后依赖。'])
for a,b,s in [(0,10,'Ea'),(10,20,'Pa'),(20,30,'Eb'),(30,40,'Da'),(40,50,'Ec'),(50,60,'Pb'),(60,70,'Ed'),(70,80,'Db'),(80,100,'后续推进')]: c.box(0,a,b,s)
for a,b,s in [(8,25,'Ea · 32行'),(28,45,'Eb · 32行'),(48,65,'Ec · 32行'),(68,85,'Ed · 32行')]: c.box(1,a,b,s)
c.box(2,0,8,'权重读')
for a,b,s in [(12,26,'Pa：本地读'),(32,46,'Da：peer读'),(52,66,'Pb：本地读'),(72,86,'Db：peer读')]: c.box(2,a,b,s)
for a,b,s in [(26,40,'Pa：回源写'),(46,60,'Da：本地写'),(66,80,'Pb：回源写'),(85,93,'激活写'),(93,100,'Db写')]: c.box(3,a,b,s)
c.save('07_aiv1_sharing.png')
print('Rendered %d figures in %s' % (len(GENERATED), OUT))
for name in GENERATED: print(name)
