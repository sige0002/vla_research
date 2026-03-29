# ViT バックボーンと VLM アーキテクチャのロボティクス応用に関する調査報告

**作成日**: 2026年3月29日

---

## 目次

1. [C-RADIO (nvidia/C-RADIOv4-SO400M)](#1-c-radio)
2. [Qwen2.5-VL / Qwen3-VL (4Bパラメータクラス)](#2-qwen25-vl--qwen3-vl)
3. [ViT バックボーン置換の実現可能性](#3-vit-バックボーン置換の実現可能性)
4. [ロボティクス向け有望な ViT バックボーン](#4-ロボティクス向け有望な-vit-バックボーン)
5. [推論速度に関する考察](#5-推論速度に関する考察)
6. [総括と提言](#6-総括と提言)

---

## 1. C-RADIO (nvidia/C-RADIOv4-SO400M)

### 1.1 概要

C-RADIO (Commercial RADIO) は NVIDIA Research が開発した「Agglomerative Vision Foundation Model」であり、複数の大規模 Vision Foundation Model を単一の学生モデルに蒸留するフレームワーク AM-RADIO の商用ライセンス版である。「RADIO」は "Reduce All Domains Into One" の略称であり、CVPR 2024 で発表された AM-RADIO を基盤として、CVPR 2025 で RADIOv2.5 が採択され、2026年1月に最新の C-RADIOv4 がリリースされた。

### 1.2 アーキテクチャ詳細

| パラメータ | C-RADIOv4-SO400M | C-RADIOv4-H |
|---|---|---|
| **総パラメータ数** | ~412M (431M) | ~631M |
| **ベースアーキテクチャ** | ViT-SO400M | ViT-H |
| **パッチサイズ** | 16 | 16 |
| **最大解像度** | 2048 px | 2048 px |
| **推奨解像度** | 512 x 512 | 512 x 512 |
| **ViTDet サポート** | あり | あり |
| **入力値範囲** | [0, 1] | [0, 1] |

**ViT-SO400M のアーキテクチャ**: "Shape-Optimized" (SO) は Alabdulmohsin et al. の "Getting ViT in Shape" 論文に基づき、スケーリング則から最適なモデル形状を予測して設計されたアーキテクチャである。SigLIP の SO400M と同系列であり、埋め込み次元は 1152 である。C-RADIO はこの ViT-SO400M をバックボーンとして使用し、パッチサイズ 16 で動作する。

**出力形式**:
- **Summary tensor**: `(B, C)` - CLSトークンに相当し、画像全体の概念を表現（分類・検索向け）
- **Spatial features**: `(B, N, C)` または `(B, C, H, W)` (NCHW形式) - 局所的な密な特徴（セグメンテーション・VLM統合向け）

**Attention 機構**: 大半の Transformer ブロックは Window Attention を使用し、一部のブロックが Global Attention を使用する。Window サイズは 6x6 ~ 32x32 トークンの範囲をサポートする。

### 1.3 学習手法

C-RADIOv4 の学習は **マルチティーチャー蒸留** に基づいている。

**教師モデル (C-RADIOv4)**:
1. **SigLIP2-g-384**: テキスト-画像アライメント（意味的理解）
2. **DINOv3-7B**: 自己教師あり学習による空間特徴（密な予測）
3. **SAM3**: セグメンテーション対応力

**蒸留の仕組み**:
- Summary トークン → SigLIP2 と DINOv3 の出力に整合
- Dense/Spatial トークン → DINOv3 と SAM3 の出力に整合
- **PHI-S (PHI Standardization)**: 各教師の空間特徴の分布を正規化し、蒸留の安定性を向上
- **DAMP**: 学習中にモデル重みに乗法的ノイズを付加し、ロバスト性を向上
- **FeatSharp**: 特徴のアップスケーリング手法（ICML 2025 採択）

**学習データ**: 約7億枚の画像（自動ラベリング、人手アノテーション不要）

**解像度学習**: {128...1152} px の範囲で確率的マルチ解像度学習を実施。

### 1.4 SigLIP / DINOv2 / SAM との比較における C-RADIO の優位性

| 特性 | SigLIP | DINOv2 | SAM | C-RADIO |
|---|---|---|---|---|
| テキストグラウンディング | 強い | 弱い | なし | **強い** (SigLIP2 アダプタ経由) |
| 密な空間特徴 | 中程度 | **強い** | 強い | **強い** (DINOv3+SAM3 蒸留) |
| セグメンテーション | 弱い | 中程度 | **強い** | **強い** (SAM3 蒸留) |
| ゼロショット分類 | 強い | 中程度 | なし | **非常に強い** (+6.8% over teachers) |
| 任意解像度対応 | 制限あり | 制限あり | 固定 | **完全対応** (最大2048px) |
| 非正方形入力 | 通常不可 | 通常不可 | 不可 | **対応** |
| 商用ライセンス | 要確認 | NC制限 | Apache | **NVIDIA Open Model License** |

**核心的な優位点**: C-RADIO は単一モデルで複数の教師の能力を統合しており、個々の教師を上回る性能を達成している。特に ImageNet ゼロショット (+6.8%)、kNN (+2.39%)、線形プロービングセグメンテーション (+3.8%)、VLM統合 (LLaVA 1.5 で最大 +1.5%) で改善を示す。

### 1.5 VLA バックボーンとしての活用可能性

C-RADIO は VLA のビジョンエンコーダとして以下の理由で有望である:

1. **統合的特徴**: セマンティック (SigLIP2) + 空間 (DINOv3) + セグメンテーション (SAM3) の三要素を単一モデルで提供
2. **アダプタ機構**: `siglip2-g`, `dino_v3`, `sam3` のアダプタを備えており、下流タスクに応じた特徴選択が可能
3. **テキストエンコーディング**: SigLIP2 アダプタを通じてテキストのトークナイズ・エンコーディングが可能であり、言語条件付き制御に直接利用可能
4. **効率的な高解像度処理**: ViTDet (Window + Global Attention) により、高解像度入力を効率的に処理
5. **NVIDIA エコシステムとの親和性**: TensorRT 等での最適化が期待される

---

## 2. Qwen2.5-VL / Qwen3-VL (4Bパラメータクラス)

### 2.1 Qwen2.5-VL アーキテクチャ

Qwen2.5-VL は Alibaba Qwen チームが開発した Vision-Language Model である。技術レポートは 2025年2月に arXiv で公開された (arXiv:2502.13923)。

#### ViT コンポーネント (Qwen2.5-VL-3B-Instruct の config.json より)

| パラメータ | 値 |
|---|---|
| **hidden_size** | 1280 |
| **num_heads** | 16 |
| **depth (レイヤ数)** | 32 |
| **intermediate_size** | 3420 |
| **patch_size** | 14 |
| **spatial_patch_size** | 14 |
| **temporal_patch_size** | 2 |
| **spatial_merge_size** | 2 |
| **window_size** | 112 (ピクセル単位、8パッチ分) |
| **hidden_act** | silu |
| **out_hidden_size** | 2048 (LLM hidden_size に合わせる) |
| **Full Attention レイヤ** | [7, 15, 23, 31] (4層) |

#### LLM コンポーネント (Qwen2.5-VL-3B)

| パラメータ | 値 |
|---|---|
| **hidden_size** | 2048 |
| **num_attention_heads** | 16 |
| **num_key_value_heads** | 2 (GQA) |
| **num_hidden_layers** | 36 |
| **intermediate_size** | 11008 |
| **max_position_embeddings** | 128,000 |

#### 主要な設計特徴

1. **ネイティブ動的解像度**: 画像の高さ・幅を 28 の倍数にリサイズし、ストライド 14 でパッチ生成。画像サイズに応じてトークン数が可変。
2. **Window Attention + Full Attention**: 大半のレイヤは Window Attention で計算効率を確保し、4層 (インデックス 7, 15, 23, 31) のみ Full Attention を使用。
3. **2D-RoPE**: 位置エンコーディングに 2D Rotary Position Embedding を採用し、可変解像度に対応。
4. **SwiGLU + RMSNorm**: ViT を LLM と同じ構造で統一。
5. **Spatial Merge**: `spatial_merge_size=2` により、隣接する 2x2 パッチを統合してトークン数を 1/4 に削減。
6. **時間的パッチ**: `temporal_patch_size=2` により、動画の連続2フレームを1つの時間パッチとして処理。
7. **絶対時間エンコーディング**: 動画フレームに実時間（秒単位）の位置エンコーディングを付与。

#### コネクタ設計

ViT の出力 (hidden_size=1280) を LLM の入力空間 (hidden_size=2048) にマッピングする `out_hidden_size=2048` の射影層が存在する。Spatial Merge 後のトークンが LLM に入力される。

### 2.2 Qwen3-VL アーキテクチャ

Qwen3-VL は 2025年9月~10月にかけて段階的にリリースされ、技術レポートは 2025年11月に公開された (arXiv:2511.21631)。Dense (2B/4B/8B/32B) と MoE (30B-A3B/235B-A22B) の両バリアントが存在する。

#### ViT コンポーネント (Qwen3-VL-4B-Instruct の config.json より)

| パラメータ | 値 |
|---|---|
| **hidden_size** | 1024 |
| **num_heads** | 16 |
| **depth (レイヤ数)** | 24 |
| **intermediate_size** | 4096 |
| **patch_size** | 16 |
| **spatial_merge_size** | 2 |
| **temporal_patch_size** | 2 |
| **hidden_act** | gelu_pytorch_tanh |
| **out_hidden_size** | 2560 (LLM hidden_size) |
| **deepstack_visual_indexes** | [5, 11, 17] |

#### LLM コンポーネント (Qwen3-VL-4B)

| パラメータ | 値 |
|---|---|
| **hidden_size** | 2560 |
| **num_attention_heads** | 32 |
| **num_key_value_heads** | 8 (GQA) |
| **num_hidden_layers** | 36 |
| **intermediate_size** | 9728 |
| **max_position_embeddings** | 262,144 |

#### Qwen2.5-VL からの主要な進化点

1. **DeepStack**: ViT の複数レイヤ (インデックス 5, 11, 17) からの特徴を LLM の複数レイヤに注入。低レベル詳細から高レベル概念まで豊富な視覚情報を保存する。
2. **Interleaved MRoPE (改良版)**: 時間・高さ・幅の情報をより均等に特徴次元に分配し、全周波数カバレッジを確保。長時間動画理解が大幅に向上。
3. **テキストベース時間アライメント**: T-RoPE から明示的テキストタイムスタンプアライメントへ進化し、時間的グラウンディング精度が向上。
4. **256Kトークンコンテキスト**: テキストとマルチモーダル入力の双方でネイティブ 256K トークンウィンドウをサポート。
5. **パッチサイズ変更**: Qwen2.5-VL の 14 から Qwen3-VL では **16** に変更。

### 2.3 Qwen2.5-VL vs Qwen3-VL (4Bクラス) 比較

| 特性 | Qwen2.5-VL-3B | Qwen3-VL-4B |
|---|---|---|
| **ViT hidden_size** | 1280 | 1024 |
| **ViT depth** | 32 | 24 |
| **ViT patch_size** | 14 | 16 |
| **LLM hidden_size** | 2048 | 2560 |
| **LLM layers** | 36 | 36 |
| **コンテキスト長** | 128K | 256K |
| **特徴注入** | 単一出力 | DeepStack (多層注入) |
| **位置エンコーディング** | 2D-RoPE (MRoPE) | Interleaved MRoPE |

注目すべきは、Qwen3-VL-4B では ViT が**小型化** (depth 32→24, hidden_size 1280→1024) されつつ、DeepStack による多層特徴注入で性能を維持・向上させている点である。これは ViT をより効率的な外部バックボーンに置き換える余地があることを示唆する。

---

## 3. ViT バックボーン置換の実現可能性

### 3.1 Qwen の ViT を C-RADIO に置換するシナリオ

#### 次元整合性の分析

| コンポーネント | Qwen2.5-VL-3B ViT | Qwen3-VL-4B ViT | C-RADIOv4-SO400M |
|---|---|---|---|
| **出力次元** | 1280 | 1024 | ~1152 (SO400M embed_dim) |
| **パッチサイズ** | 14 | 16 | 16 |
| **LLM入力次元** | 2048 | 2560 | - |

C-RADIOv4-SO400M の出力特徴次元は約 1152 であり、Qwen2.5-VL の ViT 出力 (1280) や Qwen3-VL の ViT 出力 (1024) とは異なる。このため、**射影層 (Projector) の再設計が必要**となる。

#### 必要なアーキテクチャ変更

1. **Projector の再設計**:
   - C-RADIO の spatial features (dim=1152) を LLM の hidden_size (2048 or 2560) にマッピングする新しい射影層が必要
   - 2層 MLP (Linear → GELU → Linear) が最もシンプルかつ実績のある設計
   - Spatial Merge (2x2 パッチ統合) 機構も射影層に組み込む必要あり

2. **パッチサイズの整合**:
   - C-RADIOv4 はパッチサイズ 16 → Qwen3-VL (パッチ16) と直接互換
   - Qwen2.5-VL (パッチ14) の場合は解像度調整が必要

3. **位置エンコーディング**:
   - Qwen の 2D-RoPE / MRoPE は ViT 内部に組み込まれているため、C-RADIO 使用時は別途対応が必要
   - C-RADIO は独自の位置エンコーディングを持つため、出力特徴をそのまま LLM に入力し、LLM 側の位置エンコーディングに任せるアプローチが考えられる

4. **DeepStack の代替** (Qwen3-VL の場合):
   - C-RADIO は単一の出力層からの summary + spatial features を提供
   - Qwen3-VL の DeepStack (ViT の複数中間層から LLM に注入) を再現するには、C-RADIO の中間層にフックを設ける必要がある
   - あるいは、DeepStack を諦めて単一層出力 + 強化された射影層で代替する

5. **動画処理**:
   - C-RADIO は基本的に静止画用であり、temporal_patch_size のような時間軸の処理は持たない
   - 動画入力には、フレーム毎に C-RADIO を適用し、時間的トークン統合を別途設計する必要がある

### 3.2 学習戦略

#### 推奨される段階的学習プロセス

**Stage 1: Projector のみの学習 (Alignment Pre-training)**
- C-RADIO (frozen) → Projector (trainable) → LLM (frozen)
- 画像-テキストペアデータで視覚特徴と言語空間のアライメントを学習
- 比較的少量のデータ (数百万ペア) で実施可能

**Stage 2: Projector + LLM の微調整**
- C-RADIO (frozen) → Projector (trainable) → LLM (trainable, low-lr)
- タスク固有のデータ（ロボット操作データ等）で微調整
- LoRA 等の効率的微調整手法の適用も有効

**Stage 3: (オプション) End-to-end 微調整**
- C-RADIO (trainable, very low-lr) → Projector (trainable) → LLM (trainable)
- ロボット操作データで全体を微調整
- ただし OpenVLA の知見によると、ViT の凍結解除は必須であり、ViT を凍結したままでは性能が大幅に低下する

#### 重要な知見 (OpenVLA より)

- **ViT の凍結は性能低下を招く**: OpenVLA の実験で、ビジョンエンコーダの凍結や最終層のみの微調整は性能が著しく低下することが示されている
- **「サンドイッチ微調整」**: LLM 全体の微調整なしに、ViT + Projector + LLM の一部層を微調整するアプローチが GPU メモリ効率と性能のバランスが良い
- **シーンへの適応が重要**: 視覚特徴をターゲットシーンに適応させることが成功の鍵

### 3.3 代替アプローチ: ViT 置換ではなく特徴融合

OpenVLA が DINOv2 + SigLIP の融合エンコーダ (DinoSigLIP) を使用しているように、C-RADIO を**追加の**視覚エンコーダとして組み込む方法も考えられる:

- **方法A**: Qwen の ViT + C-RADIO の特徴を連結 → 射影層で LLM 空間にマッピング
- **方法B**: C-RADIO の特徴のみを使用し、Qwen の ViT を完全に置換
- **方法C**: C-RADIO のアダプタ出力 (siglip2-g, dino_v3 等) を個別に活用

既に NVIDIA が C-RADIOv4 で SAM3 のビジョンエンコーダ置換を実証していることは注目に値する (sam3-radio デモ)。

---

## 4. ロボティクス向け有望な ViT バックボーン

### 4.1 主要なビジョンバックボーンの比較

| モデル | パラメータ | 空間特徴 | 意味的理解 | ロボティクス実績 | ライセンス |
|---|---|---|---|---|---|
| **C-RADIOv4-SO400M** | 412M | 非常に強い (DINOv3+SAM3) | 強い (SigLIP2) | 間接的 (VLM統合実績) | 商用可 |
| **SigLIP-SO400M** | 400M | 中程度 | 強い | OpenVLA, pi0 で採用 | Apache 2.0 |
| **DINOv2-g** | 1.1B | 非常に強い | 中程度 | OpenVLA で採用 (融合) | Apache 2.0 |
| **InternViT-6B** | 6B | 強い | 強い | InternVL で採用 | 商用可 |
| **EVA-CLIP-18B** | 18B | 強い | 強い | SAM 等の内部利用 | MIT |
| **Theia (ViT-B)** | 86M | 強い (蒸留) | 中程度 | CortexBench SOTA | 要確認 |

### 4.2 ロボット操作に重要な特徴

**1. 空間特徴 (Spatial Features)** - 最重要
- 物体の位置・形状・6DoF姿勢の正確な推定に不可欠
- DINOv2 の per-patch 特徴が特に高評価
- C-RADIO は DINOv3 蒸留により強力な空間特徴を提供

**2. 意味的特徴 (Semantic Features)**
- 言語指示の理解と物体認識に必要
- SigLIP / CLIP 系のテキスト-画像アライメントが重要
- C-RADIO は SigLIP2 アダプタでこれをカバー

**3. 時間的特徴 (Temporal Features)**
- 動作の予測と物理ダイナミクスの理解に重要
- 現行の静止画 ViT では本質的に弱い
- Video Foundation Model (例: Cosmos-Predict2) への移行が議論されている

### 4.3 Theia: ロボット学習特化の蒸留モデル

Boston Dynamics AI Institute の Theia は C-RADIO と類似のマルチティーチャー蒸留アプローチをロボット学習に特化して適用したモデルであり、特筆に値する。

- **教師モデル**: CLIP + DINOv2 + ViT (CDiV が最良の組み合わせ)
- **学生モデル**: ViT-Tiny/Small/Base (ロボットの限られた計算資源を考慮)
- **重要な発見**: 空間トークン表現のエントロピーとロボット学習性能に強い相関 (R=0.943) がある。エントロピーの高い特徴（特徴の多様性が高い）ほど、ポリシー学習を助ける情報をより多く符号化している。
- **性能**: 教師モデルを上回る性能を、より小さなモデルサイズとより少ない学習データで達成

### 4.4 Video Model バックボーンへの転換

Mimic Robotics の mimic-video は、VLM バックボーンを**ビデオモデルバックボーン** (NVIDIA Cosmos-Predict2) に完全に置き換えるアプローチを提案している。

**根拠**: VLM は静止画-言語データで事前学習されており、物理ダイナミクスや運動に関する知識は本質的に弱い。一方、ビデオモデルはセマンティクス、物理ダイナミクス、高レベル行動を事前学習段階で統合的にモデル化できる。

これは将来的に ViT バックボーンの選択よりも根本的なアーキテクチャ転換となりうる重要な動向である。

---

## 5. 推論速度に関する考察

### 5.1 VLA の速度要件

| ポリシーレベル | 必要周波数 | レイテンシ上限 | 用途 |
|---|---|---|---|
| **高レベルポリシー** | 5-10 Hz | 100-200 ms | タスクプランニング、粗い動作 |
| **低レベルポリシー** | 30-100 Hz | 10-33 ms | 精密制御、力制御 |

VLA-Perf (2026年2月発表) によれば、リアルタイム推論の目標は **10~100ms のレイテンシ** である。33ms 以下 (約30Hz) が、30FPS RGB ビデオストリームの全フレームを処理可能な転換点である。

### 5.2 現行 VLA モデルの推論速度

| モデル | パラメータ | GPU | 推論速度 | 備考 |
|---|---|---|---|---|
| **OpenVLA** | 7B | A5000 (INT4) | ~3 Hz | 量子化でメモリ半減 |
| **OpenVLA-OFT** | 7B | - | 25-50x 高速化 | 最適化レシピ (2025年3月) |
| **SmolVLA** | 0.45B | Consumer GPU | 高速 | pi0 比 40% 高速学習 |
| **pi0相当 (最適化)** | ~3B | Consumer GPU | 30Hz + 480Hz trajectory | 最新の最適化手法 |

### 5.3 量子化と最適化

- **INT4 量子化**: bfloat16 と同等の性能を維持しつつ、GPU メモリ使用量を半分以下に削減。Ada Lovelace アーキテクチャ (RTX 4090, H100) で特に高スループット。
- **モデルサイズ削減**: SmolVLA (0.45B) は OpenVLA (7B) や pi0 (3.3B) を上回る成功率 (LIBERO で 87.3%) を達成しており、小型モデルの有効性が実証されている。
- **非同期推論**: SmolVLA の非同期推論スタックにより、アクションチャンク間のスムーズな連続性を確保しつつ高速化を実現。
- **VLM レイヤスキッピング**: 選択的に VLM レイヤをスキップすることで計算量を削減するアプローチ。
- **Action Chunking**: 一度の推論で複数ステップのアクションを予測し、実効的な制御周波数を向上。

### 5.4 C-RADIO + Qwen 系での速度見積もり

**5Hz 高レベルポリシー (200ms budget)**:
- C-RADIOv4-SO400M (ViT forward): ~10-20ms (bfloat16, RTX 4090)
- Projector: ~1ms
- Qwen3-VL-4B LLM (アクション生成): ~50-150ms (INT4 量子化)
- **合計: ~60-170ms → 5Hz は十分に実現可能**

**30Hz 低レベルポリシー (33ms budget)**:
- 全モデルを 33ms 以内に収めるのは困難
- **Action Chunking** が必須: 5Hz で推論し、チャンクサイズ 6-20 で 30-100Hz の制御を実現
- **Fast-Slow アーキテクチャ**: 高レベル VLA (低頻度) + 低レベルリアクティブ制御器 (高頻度) の二層構造

---

## 6. 総括と提言

### 6.1 C-RADIO を VLA バックボーンとして使用する戦略的意義

C-RADIOv4-SO400M は VLA のビジョンエンコーダとして極めて有望である。その理由は:

1. **統合的表現**: 意味的理解 (SigLIP2)、空間特徴 (DINOv3)、セグメンテーション (SAM3) を単一モデルで提供し、OpenVLA の DinoSigLIP 融合エンコーダと同等以上の表現力を期待できる
2. **効率性**: 412M パラメータで複数の大型教師モデルの能力を凝縮。DINOv2-g (1.1B) + SigLIP-SO400M (400M) の合計 (1.5B) と比較して大幅に軽量
3. **任意解像度対応**: ロボットカメラの多様な解像度・アスペクト比に柔軟に対応可能
4. **商用ライセンス**: NVIDIA Open Model License により商用ロボティクス製品への組み込みが可能
5. **Theia の知見との整合**: マルチティーチャー蒸留が高エントロピーの空間特徴を生成し、ロボット学習性能と強く相関するという Theia の発見は、C-RADIO の設計哲学を直接支持する

### 6.2 推奨する実装ロードマップ

**Phase 1: 概念実証 (PoC)**
- C-RADIOv4-SO400M を frozen ViT として使用
- 2層 MLP Projector を学習
- 小規模 LLM (Qwen3-VL-4B の text 部分) と組み合わせ
- シンプルな操作タスク (pick-and-place) で検証

**Phase 2: VLA 統合**
- Action Chunking + Flow Matching (SmolVLA 方式) を採用
- C-RADIO のアダプタ出力も活用 (dino_v3 特徴で空間精度向上)
- 段階的微調整 (Projector → LLM → ViT)

**Phase 3: 最適化とデプロイ**
- INT4 量子化 + TensorRT 最適化
- Fast-Slow アーキテクチャで 5Hz VLA + 30-100Hz 低レベル制御
- エッジデバイス (Jetson Orin) でのデプロイ検証

### 6.3 リスクと注意点

1. **ViT 微調整の必要性**: OpenVLA の知見から、ViT を完全に凍結するのは性能面で不利。C-RADIO の微調整時に蒸留で獲得した特性が崩壊するリスクがある。
2. **動画理解の欠如**: C-RADIO は静止画モデルであり、temporal_patch_size のような時間軸処理を持たない。ロボットの動的環境理解には別途対策が必要。
3. **Video Model への転換**: mimic-video のようなビデオモデルバックボーンへの潮流が加速する場合、ViT ベースのアプローチ自体が陳腐化するリスクがある。
4. **DeepStack の喪失**: Qwen3-VL の DeepStack を C-RADIO で再現するのは技術的に複雑であり、性能低下の可能性がある。

---

## 参考文献・情報源

### C-RADIO 関連
- [C-RADIOv4 Tech Report (arXiv:2601.17237)](https://arxiv.org/abs/2601.17237)
- [AM-RADIO: CVPR 2024 Paper (arXiv:2312.06709)](https://arxiv.org/abs/2312.06709)
- [RADIOv2.5: CVPR 2025 Paper (arXiv:2412.07679)](https://arxiv.org/abs/2412.07679)
- [NVlabs/RADIO GitHub Repository](https://github.com/NVlabs/RADIO)
- [nvidia/C-RADIOv4-SO400M on HuggingFace](https://huggingface.co/nvidia/C-RADIOv4-SO400M)
- [C-RADIOv4 Vision Backbone (MarkTechPost)](https://www.marktechpost.com/2026/02/06/nvidia-ai-releases-c-radiov4-vision-backbone-unifying-siglip2-dinov3-sam3-for-classification-dense-prediction-segmentation-workloads-at-scale/)
- [C-RADIOv4: A Distilled Vision Foundation Model (Voxel51)](https://voxel51.com/blog/c-radiov4-distilled-vision-foundation-model)

### Qwen VL 関連
- [Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Qwen3-VL Technical Report (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [Qwen2.5-VL-3B-Instruct config.json](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/config.json)
- [Qwen3-VL-4B-Instruct config.json](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/main/config.json)
- [Qwen2.5-VL Architecture (DebuggerCafe)](https://debuggercafe.com/qwen2-5-vl/)
- [Qwen3-VL Model Architecture (DeepWiki)](https://deepwiki.com/QwenLM/Qwen3-VL/4.2-model-architecture)

### VLA / ロボティクス関連
- [OpenVLA (arXiv:2406.09246)](https://arxiv.org/abs/2406.09246)
- [SmolVLA (arXiv:2506.01844)](https://arxiv.org/abs/2506.01844)
- [VLA-Perf Benchmark (arXiv:2602.18397)](https://arxiv.org/html/2602.18397v1)
- [Theia: Distilling Diverse VFMs for Robot Learning (arXiv:2407.20179)](https://arxiv.org/abs/2407.20179)
- [Large VLM-based VLA Survey](https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation)
- [Video-Action Models (Mimic Robotics)](https://www.mimicrobotics.com/blog/video-action-models-are-video-model-backbones-the-future-of-vlas)
- [VLA Survey (Nature Machine Intelligence, 2025)](https://www.nature.com/articles/s42256-025-01168-7)
- [Helix: VLA for Humanoid Control (Figure AI)](https://www.figure.ai/news/helix)
- [VLASH: Real-Time VLAs (arXiv:2512.01031)](https://arxiv.org/html/2512.01031)
- [State of VLA at ICLR 2026](https://mbreuss.github.io/blog_post_iclr_26_vla.html)

### その他ビジョンモデル
- [FeatSharp (ICML 2025, arXiv:2502.16025)](https://www.arxiv.org/abs/2502.16025)
- [PHI-S (arXiv:2410.01680)](https://arxiv.org/abs/2410.01680)
- [SigLIP SO400M (HuggingFace)](https://huggingface.co/google/siglip-so400m-patch14-384)
- [EVA-CLIP (GitHub)](https://github.com/baaivision/EVA)
