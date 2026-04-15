import graphviz

# 初始化图表，设置为从左到右的布局
dot = graphviz.Digraph('Vision_Action_Modeling', format='pdf')
dot.attr(rankdir='LR', compound='true', splines='polyline', fontname='Helvetica')

# 设置全局节点样式 (圆角矩形，浅灰色背景)
dot.attr('node', shape='box', style='rounded,filled', fillcolor='#f8f9fa', fontname='Helvetica')

# ---------------- M1: Decoupled Modality Modeling ----------------
with dot.subgraph(name='cluster_A') as c:
    c.attr(label='(a) Decoupled Modality Modeling', style='dashed', color='gray')
    
    # Vision Pipeline
    c.node('V1', 'Vision V')
    c.node('ZV1', 'Latent Z_v')
    c.node('RV1', 'Recon V')
    c.edge('V1', 'ZV1')
    c.edge('ZV1', 'RV1')

    # Action Pipeline
    c.node('A1', 'Action A')
    c.node('ZA1', 'Latent Z_a')
    c.node('RA1', 'Recon A')
    c.edge('A1', 'ZA1')
    c.edge('ZA1', 'RA1')

# ---------------- M2: Action-Guided Vision Encoding ----------------
with dot.subgraph(name='cluster_B') as c:
    c.attr(label='(b) Action-Guided Vision Encoding', style='dashed', color='gray')
    
    c.node('V2', 'Vision V')
    c.node('ZV2', 'Latent Z_v')
    c.node('RV2', 'Recon V')
    c.edge('V2', 'ZV2')
    c.edge('ZV2', 'RV2')

    c.node('A2', 'Action A')
    # Action 作为引导信号单向输入
    c.edge('A2', 'ZV2', label=' Guidance', style='dashed', color='blue', fontcolor='blue')

# ---------------- M3: Distribution-Level Alignment ----------------
with dot.subgraph(name='cluster_C') as c:
    c.attr(label='(c) Distribution-Level Alignment', style='dashed', color='gray')
    
    c.node('V3', 'Vision V')
    c.node('ZV3', 'Latent Z_v')
    c.node('RV3', 'Recon V')
    c.edge('V3', 'ZV3')
    c.edge('ZV3', 'RV3')

    c.node('A3', 'Action A')
    c.node('ZA3', 'Latent Z_a')
    c.node('RA3', 'Recon A')
    c.edge('A3', 'ZA3')
    c.edge('ZA3', 'RA3')

    # Latent 空间分布对齐
    c.edge('ZV3', 'ZA3', dir='both', label=' KL Alignment', color='purple', style='dotted', fontcolor='purple')

# ---------------- M4: Ours: Unified Vision-Action Learning ----------------
with dot.subgraph(name='cluster_D') as c:
    # 加粗外框以突出 Ours
    c.attr(label='(d) Ours: Unified Vision-Action Modeling', style='bold', color='black', penwidth='2')
    
    # 输入端 (包含 Joint Input)
    c.node('V4', 'Vision V')
    c.node('A4', 'Action A')
    c.node('VA4', 'Vision + Action')

    # 统一空间 (高亮背景色)
    c.node('Z4', 'Unified Latent Z\n(Codebook)', fillcolor='#e6f2ff', style='filled,rounded,bold', penwidth='1.5')

    # 输出端
    c.node('RV4', 'Recon V')
    c.node('RA4', 'Recon A')
    c.node('FUSED', 'Fused V-A Feature')

    # 编码路径
    c.edge('V4', 'Z4')
    c.edge('A4', 'Z4')
    c.edge('VA4', 'Z4', label=' Joint Input')

    # 解码路径 (💥 核心 Highlight: FDM, IDM, Deep Fusion 采用红色和绿色粗线条)
    c.edge('Z4', 'RV4', label=' FDM (a2v)', color='#d62728', penwidth='2.5', fontcolor='#d62728')
    c.edge('Z4', 'RA4', label=' IDM (v2a)', color='#d62728', penwidth='2.5', fontcolor='#d62728')
    c.edge('Z4', 'FUSED', label=' Deep Fusion', color='#2ca02c', penwidth='2.5', fontcolor='#2ca02c')

    # 隐式的 v2v 和 a2a 重建 (弱化显示)
    c.edge('Z4', 'RV4', style='dotted', color='gray', label=' v2v')
    c.edge('Z4', 'RA4', style='dotted', color='gray', label=' a2a')

# 渲染并保存为 PDF (不自动打开)
output_path = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/Figure1_Method_Comparison'
dot.render(output_path, view=False)
print(f"学术对比图已生成: {output_path}.pdf")