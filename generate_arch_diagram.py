"""
Генерация красивой диаграммы архитектуры DualPolMAFUformerMIL.
"""

import graphviz

def create_architecture_diagram():
    g = graphviz.Digraph(
        "DualPolMAFUformerMIL",
        format="png",
        engine="dot",
        graph_attr={
            "rankdir": "TB",
            "bgcolor": "#0d1117",
            "fontname": "Helvetica",
            "fontsize": "14",
            "fontcolor": "white",
            "pad": "0.5",
            "nodesep": "0.4",
            "ranksep": "0.6",
            "dpi": "150",
            "label": "DualPolMAFUformerMIL Architecture",
            "labelloc": "t",
            "labeljust": "c",
            "fontsize": "22",
            "fontcolor": "#58a6ff",
        },
        node_attr={
            "fontname": "Helvetica",
            "fontsize": "11",
            "style": "filled,rounded",
            "shape": "box",
            "penwidth": "0",
            "margin": "0.15,0.08",
        },
        edge_attr={
            "color": "#484f58",
            "arrowsize": "0.7",
            "penwidth": "1.5",
        },
    )

    # ── Цветовая палитра ──
    INPUT    = {"fillcolor": "#1f6feb", "fontcolor": "white"}
    STEM     = {"fillcolor": "#238636", "fontcolor": "white"}
    FUSION   = {"fillcolor": "#da3633", "fontcolor": "white"}
    BACKBONE = {"fillcolor": "#8957e5", "fontcolor": "white"}
    BOTTLE   = {"fillcolor": "#6e40c9", "fontcolor": "white"}
    DECODER  = {"fillcolor": "#d29922", "fontcolor": "#0d1117"}
    SEG      = {"fillcolor": "#f78166", "fontcolor": "#0d1117"}
    EMBED    = {"fillcolor": "#3fb950", "fontcolor": "#0d1117"}
    MIL      = {"fillcolor": "#58a6ff", "fontcolor": "#0d1117"}
    OUTPUT   = {"fillcolor": "#f0f6fc", "fontcolor": "#0d1117", "penwidth": "2", "color": "#58a6ff"}
    DS       = {"fillcolor": "#484f58", "fontcolor": "#c9d1d9"}
    AUG      = {"fillcolor": "#21262d", "fontcolor": "#8b949e", "style": "filled,rounded,dashed"}

    # ── Входы ──
    with g.subgraph(name="cluster_input") as c:
        c.attr(label="Input (per tile: 512×512)", style="dashed", color="#484f58",
               fontcolor="#8b949e", fontsize="10")
        c.node("vv", "VV channel\n(1×512×512)", **INPUT)
        c.node("vh", "VH channel\n(1×512×512)", **INPUT)

    # ── Stems ──
    with g.subgraph(name="cluster_stems") as c:
        c.attr(label="1. Dual-Pol Stems (stride 4×)", style="dashed",
               color="#484f58", fontcolor="#8b949e", fontsize="10")
        c.node("stem_vv", "Stem VV\nConv3×3 s2 → Conv3×3 s2\n1→32→48ch", **STEM)
        c.node("stem_vh", "Stem VH\nConv3×3 s2 → Conv3×3 s2\n1→32→48ch", **STEM)

    g.edge("vv", "stem_vv")
    g.edge("vh", "stem_vh")

    # ── Fusion ──
    g.node("fusion", "2. Polarimetric Fusion\ngate = σ(conv(cat(Fvv, Fvh, |Fvv−Fvh|)))\nF0 = conv(gate·Fvv + (1−gate)·Fvh)\n48→96ch @ 128×128", **FUSION)
    g.edge("stem_vv", "fusion")
    g.edge("stem_vh", "fusion")

    # ── Backbone ──
    with g.subgraph(name="cluster_backbone") as c:
        c.attr(label="3. ConvNeXt-Small Backbone", style="dashed",
               color="#484f58", fontcolor="#8b949e", fontsize="10")
        c.node("adapt", "Stem Adapter\nDWConv3×3 + Conv1×1\n96→96ch", **BACKBONE)
        c.node("s0", "Stage 0\n96ch @ 128×128", **BACKBONE)
        c.node("s1", "Stage 1\n192ch @ 64×64", **BACKBONE)
        c.node("s2", "Stage 2\n384ch @ 32×32", **BACKBONE)
        c.node("s3", "Stage 3\n768ch @ 16×16", **BACKBONE)

    g.edge("fusion", "adapt")
    g.edge("adapt", "s0")
    g.edge("s0", "s1")
    g.edge("s1", "s2")
    g.edge("s2", "s3")

    # ── Bottleneck ──
    g.node("bottle", "4. Bottleneck\n2× (Window SA [8h, w=7] + DW-Conv MLP)\n768ch @ 16×16", **BOTTLE)
    g.edge("s3", "bottle")

    # ── Decoder ──
    with g.subgraph(name="cluster_decoder") as c:
        c.attr(label="5. U-Decoder + SAFB + Deep Supervision", style="dashed",
               color="#484f58", fontcolor="#8b949e", fontsize="10")
        c.node("d0", "SAFB Level 0\nskip: Stage2 (384ch)\n768→384ch @ 32×32", **DECODER)
        c.node("d1", "SAFB Level 1\nskip: Stage1 (192ch)\n384→192ch @ 64×64", **DECODER)
        c.node("d2", "SAFB Level 2\nskip: Stage0 (96ch)\n192→96ch @ 128×128", **DECODER)
        # Deep supervision
        c.node("ds0", "Aux head 0\n(w=0.25)", **DS)
        c.node("ds1", "Aux head 1\n(w=0.5)", **DS)
        c.node("ds2", "Aux head 2\n(w=1.0)", **DS)

    g.edge("bottle", "d0")
    g.edge("s2", "d0", style="dashed", color="#d29922", label="skip", fontcolor="#d29922", fontsize="9")
    g.edge("d0", "d1")
    g.edge("s1", "d1", style="dashed", color="#d29922", label="skip", fontcolor="#d29922", fontsize="9")
    g.edge("d1", "d2")
    g.edge("s0", "d2", style="dashed", color="#d29922", label="skip", fontcolor="#d29922", fontsize="9")

    g.edge("d0", "ds0", style="dotted", color="#484f58")
    g.edge("d1", "ds1", style="dotted", color="#484f58")
    g.edge("d2", "ds2", style="dotted", color="#484f58")

    # ── Multi-scale Seg Head ──
    g.node("seg_ms", "6. Multi-Scale Seg Head\nfuse(96ch + 384ch↑ + 192ch↑) → 96ch\nConv3×3 → GELU → Conv1×1 → 1ch\nupsample 4× → 512×512", **SEG)
    g.edge("d2", "seg_ms")
    g.edge("d0", "seg_ms", style="dashed", color="#f78166", label="multi-scale", fontcolor="#f78166", fontsize="9")
    g.edge("d1", "seg_ms", style="dashed", color="#f78166", fontsize="9")

    # ── Output mask ──
    g.node("mask_out", "Oil Mask Logits\n(1×512×512)", **OUTPUT)
    g.edge("seg_ms", "mask_out")

    # ── Tile Embedding ──
    g.node("tile_emb", "7. Tile Embedding\nz1=GeM(Stage3) + z2=GAP(decoder)\n+ mask_stats [mean,max,std,pos_ratio]\nMLP → 512d", **EMBED)
    g.edge("s3", "tile_emb", style="dashed", color="#3fb950", label="GeM", fontcolor="#3fb950", fontsize="9")
    g.edge("d2", "tile_emb", style="dashed", color="#3fb950", label="GAP", fontcolor="#3fb950", fontsize="9")
    g.edge("mask_out", "tile_emb", style="dashed", color="#3fb950", label="stats", fontcolor="#3fb950", fontsize="9")

    # ── MIL Bag Classifier ──
    g.node("bag_note", "×16 tiles per image (4×4 grid)", shape="plaintext",
           fontcolor="#8b949e", fontsize="10")
    g.node("mil", "8. MIL Bag Classifier\n2-layer Transformer (512d, 8 heads)\n+ CLS token + pos embed\nMLP: 512→256→3", **MIL)
    g.edge("tile_emb", "bag_note", style="invis")
    g.edge("bag_note", "mil", style="dotted", color="#484f58")
    g.edge("tile_emb", "mil")

    # ── Output class ──
    g.node("cls_out", "Class Logits\noil | lookalike | no_oil", **OUTPUT)
    g.edge("mil", "cls_out")

    # ── Loss annotation ──
    g.node("loss_seg", "L_seg = 0.5·Focal + 0.3·Dice + 0.2·Boundary\n(OHEM top 70%)",
           shape="plaintext", fontcolor="#f78166", fontsize="10")
    g.node("loss_cls", "L_cls = CrossEntropy\n(label_smoothing, class_weights)",
           shape="plaintext", fontcolor="#58a6ff", fontsize="10")
    g.node("loss_ds", "L_ds = Σ wᵢ · SegLoss(auxᵢ)",
           shape="plaintext", fontcolor="#484f58", fontsize="10")

    g.edge("mask_out", "loss_seg", style="dotted", color="#f78166", arrowhead="none")
    g.edge("cls_out", "loss_cls", style="dotted", color="#58a6ff", arrowhead="none")
    g.edge("ds2", "loss_ds", style="dotted", color="#484f58", arrowhead="none")

    return g


if __name__ == "__main__":
    g = create_architecture_diagram()
    out_path = g.render("architecture", directory=".", cleanup=True)
    print(f"Saved: {out_path}")
