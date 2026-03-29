# Qwen3.5-4B VLA統合に向けた技術調査レポート

**作成日**: 2026年3月29日
**対象モデル**: Qwen3.5-4B (マルチモーダル版)
**目的**: Vision-Language-Action (VLA) モデルへの統合可能性の技術的評価

---

## 目次

1. [Qwen3.5-4Bアーキテクチャの詳細分析](#1-qwen35-4bアーキテクチャの詳細分析)
2. [Gated DeltaNet: 線形注意機構の技術的解明](#2-gated-deltanet-線形注意機構の技術的解明)
3. [Vision Encoder (ViT) の構成と交換可能性](#3-vision-encoder-vit-の構成と交換可能性)
4. [Multi-Token Prediction (MTP) とロボットアクション生成への応用](#4-multi-token-prediction-mtp-とロボットアクション生成への応用)
5. [推論レイテンシとハードウェアベンチマーク](#5-推論レイテンシとハードウェアベンチマーク)
6. [量子化 (INT4/AWQ) とLoRAファインチューニング](#6-量子化-int4awq-とloraファインチューニング)
7. [VLA統合の現状: StarVLAおよび先行研究](#7-vla統合の現状-starvlaおよび先行研究)
8. [線形注意/SSMベースVLAとの比較分析](#8-線形注意ssmベースvlaとの比較分析)
9. [Qwen3.5 vs Qwen3-VL: 性能比較](#9-qwen35-vs-qwen3-vl-性能比較)
10. [VLA統合に向けた技術的考察と課題](#10-vla統合に向けた技術的考察と課題)

---

## 1. Qwen3.5-4Bアーキテクチャの詳細分析

### 1.1 config.jsonから得られた正確なアーキテクチャ仕様

HuggingFace (`Qwen/Qwen3.5-4B`) から取得したconfig.jsonに基づく正確な仕様:

| パラメータ | 値 |
|---|---|
| アーキテクチャクラス | `Qwen3_5ForConditionalGeneration` |
| model_type | `qwen3_5` |
| 総レイヤー数 | 32 |
| hidden_size | 2560 |
| intermediate_size (FFN) | 9216 |
| num_attention_heads | 16 |
| num_key_value_heads (GQA) | 4 |
| head_dim | 256 |
| 最大コンテキスト長 | 262,144 トークン |
| vocab_size | 248,320 |
| 活性化関数 | SiLU |
| RMSNorm epsilon | 1e-06 |
| tie_word_embeddings | true |
| dtype | bfloat16 |

### 1.2 ハイブリッド注意レイヤー構成

**layer_types配列の完全な構成** (32層):

```
[linear, linear, linear, full,    # Block 1 (Layer 0-3)
 linear, linear, linear, full,    # Block 2 (Layer 4-7)
 linear, linear, linear, full,    # Block 3 (Layer 8-11)
 linear, linear, linear, full,    # Block 4 (Layer 12-15)
 linear, linear, linear, full,    # Block 5 (Layer 16-19)
 linear, linear, linear, full,    # Block 6 (Layer 20-23)
 linear, linear, linear, full,    # Block 7 (Layer 24-27)
 linear, linear, linear, full]    # Block 8 (Layer 28-31)
```

- **線形注意 (Gated DeltaNet) レイヤー**: 24層 (75%)
- **フルソフトマックス注意レイヤー**: 8層 (25%)
- **比率**: 3:1 (Kimi Linearと同一の設計方針)

### 1.3 線形注意レイヤー固有のパラメータ

| パラメータ | 値 |
|---|---|
| linear_key_head_dim | 128 |
| linear_value_head_dim | 128 |
| linear_num_key_heads | 16 |
| linear_num_value_heads | 32 |
| linear_conv_kernel_dim | 4 |
| attn_output_gate | true |
| full_attention_interval | 4 |

**注目すべき設計特徴**:
- Key heads (16) と Value heads (32) の非対称構成。Valueヘッド数がKeyヘッド数の2倍であり、これは再帰状態行列 `S` の表現力を高めるための設計と推測される
- Causal depthwise convolution (kernel=4) がRoPEの代わりに局所的な位置情報を提供
- `partial_rotary_factor: 0.25` により、RoPEは次元の25%のみに適用

### 1.4 フルアテンションレイヤーの構成

- GQA: 16 query heads, 4 KV heads (4:1比率)
- Head dim: 256
- RoPE: `mrope_interleaved: true`, section=[11, 11, 10] (マルチモーダルRoPE)
- `rope_theta: 10,000,000` (1e7、長文脈対応)

### 1.5 Qwen3-4Bとの主要な差異

| 項目 | Qwen3-4B | Qwen3.5-4B |
|---|---|---|
| アーキテクチャ | Qwen3ForCausalLM (テキストのみ) | Qwen3_5ForConditionalGeneration (マルチモーダル) |
| レイヤー数 | 36 | 32 |
| 注意メカニズム | 全層フルソフトマックス | ハイブリッド (75% Gated DeltaNet) |
| head_dim | 128 | 256 |
| intermediate_size | 9728 | 9216 |
| num_key_value_heads | 8 | 4 (フルアテンション) / 16 (線形) |
| max_position_embeddings | 40,960 | 262,144 |
| vocab_size | 151,936 | 248,320 |
| ViT統合 | なし | あり |
| MTP | なし | あり (mtp_num_hidden_layers=1) |

---

## 2. Gated DeltaNet: 線形注意機構の技術的解明

### 2.1 理論的背景

Gated DeltaNetはICLR 2025で採択された論文「Gated Delta Networks: Improving Mamba2 with Delta Rule」(arXiv:2412.06464) に基づく。Mamba2のゲーティング機構とDelta Rule（連想記憶の誤り訂正更新規則）を統合した線形注意の変種である。

### 2.2 数学的定式化

推論時の再帰的状態更新:

```
# 状態減衰 (Memory Decay)
S_t = g_t * S_{t-1}

# 記憶からの予測
kv_mem_t = (S_t * k_t).sum()

# デルタルール: 予測誤差の計算
Delta_t = (v_t - kv_mem_t) * beta_t

# 状態更新
S_t = S_t + k_t * Delta_t

# 出力計算
y_t = (S_t * q_t).sum()
```

ここで:
- `S_t`: 再帰的状態行列 (batch, num_heads, key_dim, value_dim) — **固定サイズ**
- `g_t` (decay gate, alpha): 記憶の減衰率を制御。`alpha_log = -A_log.exp() * softplus(W_alpha(x) + dt_bias)` として計算
- `beta_t` (update gate): 新しい入力が状態をどの程度変更するかを制御。`beta = sigmoid(W_beta(x))`
- `k_t`, `q_t`: L2正規化されたKey/Query（ソフトマックスの代替）

### 2.3 デルタルールの本質

従来の線形注意（例: Linear Transformer, RetNet）との根本的な違いは**誤り訂正**にある:

1. **従来の線形注意**: `S_t = g_t * S_{t-1} + k_t * v_t` — 単純な加算更新。古い情報が新しい情報と干渉する
2. **DeltaNet**: `S_t = g_t * S_{t-1} + k_t * (v_t - S_{t-1}^T * k_t) * beta_t` — 記憶に既に存在する情報との**差分**のみを更新

この誤り訂正機構により:
- In-context retrieval能力が大幅に向上
- 連想記憶の飽和問題を回避
- Mamba2やRetNetを超える長文脈理解性能

### 2.4 ゲーティング機構の詳細

3種類のゲートが協調動作する:

1. **Decay Gate (alpha)**: 指数減衰による適応的忘却。ヘッドごとにデータ依存で減衰率が異なる
2. **Update Gate (beta)**: シグモイド関数により更新強度を [0,1] に制限。不要な更新を抑制
3. **Output Gate**: SiLU活性化 + RMSNormの後に適用。`output * silu(W_gate(x))`。Attention SinkやMassive Activationの問題を解消

### 2.5 ファインチューニング安定性への含意

**肯定的側面**:
- Output Gatingにより、Attention Sink問題（特定トークンへの注意の集中）が解消され、学習が安定
- L2正規化されたQ/Kにより、勾配のスケールが安定
- SiLU出力ゲートが勾配フローを改善（シグモイドより好ましい特性）
- 固定サイズの再帰状態は、長系列でのメモリ効率が良好

**懸念事項**:
- ハイブリッドアーキテクチャでは、線形注意層とフルアテンション層で学習率の最適値が異なる可能性
- `mamba_ssm_dtype: float32` がconfig.jsonに存在 — SSM関連の計算はfloat32精度が必要な場合がある
- 量子化時、注意層（特にフルアテンション層）を量子化するとハイブリッドアーキテクチャの性能が急激に劣化する報告あり

### 2.6 計算量比較

| 方式 | Prefill計算量 | デコード計算量/トークン | KVキャッシュ |
|---|---|---|---|
| フルソフトマックス注意 | O(n^2 * d) | O(n * d) | O(n * d) — 系列長に比例 |
| Gated DeltaNet | O(n * d^2) チャンク並列 | O(d^2) | O(d^2) — **固定サイズ** |

ロボット制御におけるVLA推論では、デコード時の一定メモリ消費は極めて重要な利点である。

---

## 3. Vision Encoder (ViT) の構成と交換可能性

### 3.1 ViTの正確な仕様 (config.jsonより)

| パラメータ | 値 |
|---|---|
| model_type | qwen3_5 (カスタムViT) |
| hidden_size | 1024 |
| intermediate_size | 4096 |
| num_heads | 16 |
| depth (レイヤー数) | 24 |
| patch_size | 16 |
| temporal_patch_size | 2 |
| spatial_merge_size | 2 |
| in_channels | 3 |
| hidden_act | gelu_pytorch_tanh |
| out_hidden_size | 2560 (= LLMのhidden_size) |
| num_position_embeddings | 2304 |
| deepstack_visual_indexes | [] (空) |

**重要な観察**:
- `out_hidden_size: 2560` がLLMの `hidden_size: 2560` と一致 — ViT出力がプロジェクタを介してLLMに直接接続
- `temporal_patch_size: 2` — 動画対応（2フレーム単位のパッチ化）
- `spatial_merge_size: 2` — 2x2の空間パッチをマージして視覚トークン数を削減

### 3.2 Qwen3-VLとの関係

Qwen3-VLでは SigLIP-2 ベースのViTが使用されており、2層MLPプロジェクタで視覚パッチをLLM次元に投影する構成をとる。Qwen3.5ではこの構成を引き継ぎつつ、ネイティブマルチモーダルモデルとして統合されている。

### 3.3 ViT交換の可能性と制約

**Qwen公式の見解** (GitHub Issue #114, QwenLM/Qwen3-VL):

> 「コードとチェックポイントを変更して試すことはできますが、お勧めしません。動的解像度をサポートするViTとLLMのアライメントに多大な労力を費やしており、別のViTに切り替えると大幅な性能低下を引き起こす可能性があります。」 — Wang Peng (Qwenチームコントリビュータ)

**技術的な交換要件**:
1. 出力次元を `out_hidden_size: 2560` に一致させる必要あり（またはプロジェクタの再設計）
2. 動的解像度対応の仕組みとの整合性
3. `spatial_merge_size: 2` のマージ処理との互換性
4. マルチモーダルRoPE (`mrope_section: [11, 11, 10]`) との位置エンコーディング整合性

**VLA用途での実践的アプローチ**:
- ViT交換よりも、既存ViTをフリーズしてアダプタ層を追加する方が現実的
- StarVLAでは、Qwen3.5のViTをそのまま活用しつつVLAヘッドを追加するアプローチを採用
- DINOv2等の深度推定に強いViTへの交換は、大規模な再学習が必要

---

## 4. Multi-Token Prediction (MTP) とロボットアクション生成への応用

### 4.1 MTPアーキテクチャ (config.jsonより)

```json
"mtp_num_hidden_layers": 1,
"mtp_use_dedicated_embeddings": false
```

- MTPヘッドは1層のTransformerブロック（Attention + FFN）で構成
- 専用の埋め込みは使用せず、メインモデルの埋め込みを共有
- 次の次のトークン（next-next token）を予測するspeculative decodingに利用可能

### 4.2 vLLMでのMTPサポート

vLLMでは `--speculative-config '{"method": "mtp", "num_speculative_tokens": N}'` で有効化可能。

**FastMTP最適化** (llama.cpp PR #20700):
- 語彙トリミング: 248K → 32Kトークンに削減 (`ggml_view_2d` でlm_headをビュー)
- ドラフト生成: 22ms → 6ms (3.7倍高速化)
- RTX 5060 Tiで約28 tok/sを達成

### 4.3 ロボットアクション生成への応用可能性

MTPはVLAにおいて以下の潜在的利点を持つ:

**1. アクションチャンキングの加速**:
ロボット制御では、1ステップで複数の将来アクションを予測する「アクションチャンキング」が主流。MTPの「複数トークン同時予測」はこの要件と自然に整合する。

**2. 低レイテンシ推論**:
- 従来: 7次元アクション × Nステップ = 7N回の自己回帰デコード
- MTP活用: speculative decodingにより、アクセプタンスレート90%で出力スループットがほぼ2倍
- B200 GPUでのvLLMベンチマークでは、MTPなしではGPU利用率が0%まで低下するケースがMTPで解消

**3. π0-FASTスタイルとの統合**:
StarVLAの `Qwen-FAST` フレームワークは、離散アクショントークンの自己回帰生成にfast tokenizerを使用。MTPによるspeculative decodingと組み合わせることで、アクション生成の制御周波数を向上させられる可能性がある。

**4. 課題**:
- MTPヘッドが1層のみであるため、アクション空間の複雑な時間的依存関係を捉えるには限界がある可能性
- VLAファインチューニング時にMTPヘッドの学習が安定するかは未検証
- アクショントークンの分布はテキストトークンと大きく異なるため、MTPのアクセプタンスレートが低下する懸念

---

## 5. 推論レイテンシとハードウェアベンチマーク

### 5.1 Qwen3.5-4Bの推定性能

直接的なQwen3.5-4Bのベンチマークデータは限定的だが、以下の情報から推定可能:

**RTX 4090 (24GB VRAM)**:
- 8Bモデルで約128 tok/s（一般的なTransformerモデル）
- Qwen3 30B-A3B (MoE) で約196 tok/s
- Qwen3.5-4Bは4Bパラメータ + 75%線形注意により、**150-200+ tok/s（BF16）** が期待される
- 75%の層がGated DeltaNet（固定サイズ状態、KVキャッシュ不要）であるため、長系列でのスループット低下が軽微

**H100 (80GB VRAM)**:
- Qwen3.5-27B BF16でのスループットデータは存在するが、4Bモデル単体の公開ベンチマークは確認できず
- HBM3帯域幅の恩恵で、4Bモデルは極めて高速な推論が可能

**VRAM要件の推定**:
- BF16: 約8GB (パラメータ) + KVキャッシュ
- Gated DeltaNet層のKVキャッシュは固定サイズ: (batch, 16, 128, 128) per head — フルアテンション層のみ系列長に比例するKVキャッシュが必要
- 8層のフルアテンション層のみがKVキャッシュを成長させるため、全層フルアテンションモデル比で**約75%のKVキャッシュ削減**

### 5.2 ロボット制御に必要な制御周波数

| アプリケーション | 必要制御周波数 | 必要レイテンシ |
|---|---|---|
| マニピュレーション（一般） | 10-30 Hz | 33-100 ms |
| 高速マニピュレーション | 50-100 Hz | 10-20 ms |
| モバイルナビゲーション | 5-10 Hz | 100-200 ms |
| ドローン制御 | 50-200 Hz | 5-20 ms |

Qwen3.5-4Bが RTX 4090 で150+ tok/sを達成する場合、1トークンあたり約6.7ms。7次元アクション出力に7トークン必要と仮定すると、約47ms（約21 Hz）でアクションチャンクを生成可能。MTP活用でこれをさらに半減できる可能性がある。

### 5.3 Gated DeltaNetの推論効率の定性的優位性

長時間の連続制御タスク（例: 数分間の組立作業）では:
- フルアテンションモデル: KVキャッシュが肥大化し、推論速度が漸次低下
- Qwen3.5: 75%のレイヤーが固定状態 → 長時間タスクでも推論速度がほぼ一定
- これはVLAにおいて極めて重要な特性

---

## 6. 量子化 (INT4/AWQ) とLoRAファインチューニング

### 6.1 量子化の現状と課題

**INT4 AWQ量子化の既知問題**:
- Qwen3-Next-80B-A3Bで、INT4 AWQ量子化後に`RuntimeError: probability tensor contains inf/nan`が発生する事例が報告 (GitHub Issue #850, NVIDIA/Model-Optimizer)
- キャリブレーションは成功するが、量子化後の推論で数値不安定性

**ハイブリッドアーキテクチャ固有の量子化戦略**:
- **重要**: アテンション層 (`attn_*`) の量子化は避けるべき — 特にハイブリッドアーキテクチャではフルアテンション層の精度が全体性能に大きく影響
- 公式INT4量子化がうまく機能する理由: アテンション層を非量子化のまま維持している
- 4bit量子化では全ての手法で顕著な性能低下が観察される (Qwen3-8B-Baseで AWQ PPL: 10.4 → 23.8)

**推奨量子化構成**:
1. FP8量子化: 性能低下が最小で、H100のFP8テンソルコアを活用可能
2. INT4 (GPTQ/AWQ): アテンション層を除外して量子化。VRAM削減と性能のバランス
3. 3bit以下: Qwen系列はLLaMA3比で低ビット量子化での性能低下が顕著 — 非推奨

### 6.2 LoRA/QLoRAファインチューニング

**QLoRAによる効率的ファインチューニング**:
- 4bit量子化ベースモデルにLoRAアダプタを追加
- DataCampによるチュートリアルが公開済み（ニュース分類タスク）
- Unslothが Qwen3.5 の LoRA/QLoRA をサポート

**BF16 LoRAの実績**:
- Qwen3.5-35B-A3B (MoEモデル) のBF16 LoRAファインチューニングがDGX Sparkで量子化なしに実現 — 4Bモデルではさらに容易

**VLA向けファインチューニング時の考慮事項**:
1. 線形注意層とフルアテンション層でLoRAランクを変えるべきか検討が必要
2. ViTのフリーズ/アンフリーズ戦略: Qwen公式はViTの再学習を推奨しない
3. アクションヘッド（MLP/Flow Matching）は full precision で学習すべき
4. `mamba_ssm_dtype: float32` の存在は、SSM関連パスでの混合精度学習に注意が必要であることを示唆

---

## 7. VLA統合の現状: StarVLAおよび先行研究

### 7.1 StarVLAにおけるQwen3.5統合

StarVLA (https://github.com/starVLA/starVLA) は、VLAモデル開発のためのモジュラーフレームワークで、2026年3月にQwen3.5を「コミュニティ最速で統合」した。

**サポートされるQwen3.5サイズ**: 0.8B, 2B, 4B, 9B

**4つのVLAフレームワーク**:

| フレームワーク | アクション生成方式 | 特徴 |
|---|---|---|
| **Qwen-FAST** | 離散アクショントークンの自己回帰生成 (π0-FAST方式) | Fast Tokenizer使用。MTPとの親和性が最も高い |
| **Qwen-OFT** | MLPヘッドによる連続アクション並列デコード (OpenVLA方式) | 1回のフォワードパスで全アクション出力 |
| **Qwen-PI** | Flow Matchingによる拡散ベースアクション予測 | 高精度だが推論が遅い |
| **Qwen-GR00T** | VL推論 + 高速アクション予測のデュアルシステム | NVIDIA GR00T方式 |

**評価ベンチマーク**: SimplerEnv, RoboCasa, LIBERO, CALVIN

### 7.2 Qwen3-VLベースのVLA研究

Medium記事 (Anton Maltsev, 2026年2月) によると、Qwen 3 VL 2Bモデル（ロボティクス専用モデルではない）を最小限のデータ（35データポイント、15分の学習）でロボットコントローラに変換し、60-70%のタスク完了率を達成。

### 7.3 LingBot-VLA

Ant Groupが2026年1月にリリースした実世界ロボットマニピュレーション向けVLA基盤モデル。LIBEROベンチマークでStarVLA比1.5-2.8倍のスピードアップを達成。

---

## 8. 線形注意/SSMベースVLAとの比較分析

### 8.1 RoboMamba (NeurIPS 2024)

- Mamba LLMと視覚エンコーダを統合したVLAモデル
- 線形計算量でのロボット推論と操作
- A100 GPU上で量子化や推論加速なしに**最高制御周波数**を達成
- 視覚的常識と空間推論能力を併せ持つ

### 8.2 SpatialVLA-Mamba

3つの革新:
1. **空間認識エンコーダ**: RGB特徴量に深度・幾何プリミティブを付加。センチメートル精度の空間グラウンディング
2. **Mambaベースデコーダ**: Transformerを置換。線形時間計算量。長期行動系列での安定したモデリング
3. **CoT-RL (Chain-of-Thought RL)**: 内在的自己改善ループ

**Qwen3.5との比較で重要な示唆**:
- Qwen3.5のGated DeltaNetはMamba2からの発展系であり、SpatialVLA-Mambaと類似の効率性を持つ
- ただし、Qwen3.5は25%のフルアテンション層を保持しており、グローバルコンテキスト理解はMambaベースモデルより優位
- Qwen3.5のViTがSpatialVLA-Mambaの空間認識エンコーダほどの深度理解を持つかは未検証

### 8.3 アーキテクチャ比較表

| モデル | バックボーン | 注意方式 | 計算量 | VLA方式 |
|---|---|---|---|---|
| OpenVLA | Llama系 | フルアテンション | O(n^2) | MLP head |
| RoboMamba | Mamba | SSM (線形) | O(n) | End-to-end |
| SpatialVLA-Mamba | Mamba + 空間エンコーダ | SSM (線形) | O(n) | CoT-RL |
| π0 | PaliGemma系 | フルアテンション | O(n^2) | Flow Matching |
| **Qwen3.5-4B VLA** | **Qwen3.5** | **ハイブリッド (75%線形)** | **O(n) 優位** | **StarVLA対応** |

---

## 9. Qwen3.5 vs Qwen3-VL: 性能比較

### 9.1 ベンチマーク比較

| ベンチマーク | Qwen3.5 | Qwen3-VL | 向上幅 |
|---|---|---|---|
| MMMU | 85.0 | 80.6 | +4.4 |
| MathVision | 88.6 | — | — |
| MMMU-Pro (9B vs 8B) | 69.2% | 56.6% | +12.6 |
| MMMU-Pro (4B vs 4B) | 65.4% | 52.0% | +13.4 |
| OmniDocBench | 90.8 | — | — |
| ERQA (Embodied Reasoning) | 67.5 | 52.5 | +15.0 |

**特筆すべき結果**:
- **ERQA (Embodied Reasoning QA)** でQwen3-VL比 +15.0ポイント — VLAにとって直接的に重要な能力
- MMMU-Proでは4Bサイズ同士で13.4ポイントの大幅改善
- テキストのみのタスクでもQwen3と同等以上 — 統一アーキテクチャの恩恵

### 9.2 効率性の比較

- Qwen3.5 397B-A17Bは長文脈タスクで**19倍高速**なデコード
- 標準ワークフローで**8.6倍高速**
- これは主にGated DeltaNetの固定状態メモリによる恩恵

---

## 10. VLA統合に向けた技術的考察と課題

### 10.1 Qwen3.5-4BをVLAバックボーンとして採用する利点

1. **推論効率**: 75%のGated DeltaNet層による線形計算量と固定KVキャッシュ。ロボットの長時間連続制御に適合
2. **マルチモーダルネイティブ**: ViT統合済みのアーキテクチャにより、画像/動画入力からのアクション生成が自然
3. **MTPによるアクション加速**: speculative decodingでアクショントークン生成を高速化可能
4. **Embodied Reasoning性能**: ERQA 67.5%はQwen3-VL比+15ポイント — 実世界環境の理解力
5. **StarVLAの即座の利用可能性**: 4Bモデルが既にサポートされ、4種のVLAフレームワークが利用可能
6. **262Kコンテキスト**: 長時間の行動履歴やマルチビュー画像を保持可能

### 10.2 主要な技術的課題

**A. Gated DeltaNetのファインチューニング安定性**:
- デルタルールの誤り訂正機構は、アクション分布学習時にどう振る舞うか未知
- `mamba_ssm_dtype: float32` が混合精度学習での問題を示唆
- LoRAを線形注意層に適用する際の最適なターゲットモジュール選定が未確立

**B. ViTの制約**:
- patch_size=16, spatial_merge_size=2 による有効解像度の制限
- ロボット用途で重要な深度推定や3D理解の追加にはアダプタ設計が必要
- ViT交換はQwen公式が非推奨 — アダプタ追加が現実的

**C. アクションヘッド設計**:
- Qwen-FAST (離散トークン) vs Qwen-OFT (連続MLP) vs Qwen-PI (Flow Matching) の最適選択
- MTPとの互換性を考慮するとQwen-FASTが自然だが、連続アクション空間の表現力に課題
- Flow Matchingは高精度だが推論が遅く、リアルタイム制御に不適

**D. 量子化とデプロイメント**:
- INT4量子化時にアテンション層の非量子化が必要 — ハイブリッドアーキテクチャでは精度管理が複雑
- RTX 4090 (24GB) でBF16実行が可能だが、バッチ推論やマルチカメラ入力時のVRAM圧迫
- Jetson Orin等のエッジデバイスへの展開にはINT4が必須だが性能低下リスク

### 10.3 推奨研究方針

**短期 (1-3ヶ月)**:
1. StarVLAの `Qwen-FAST` フレームワークでQwen3.5-4Bを使用し、LIBEROベンチマークでの基礎評価を実施
2. LoRA + BF16でのファインチューニング安定性を検証（線形注意層 vs フルアテンション層でのLoRA効果比較）
3. MTPを有効化したアクション生成の制御周波数測定

**中期 (3-6ヶ月)**:
1. Gated DeltaNet層の再帰状態がロボットのタスク文脈をどの程度保持するか定量評価
2. 深度推定アダプタの追加とSpatialVLA的な空間グラウンディングの実現
3. INT4量子化 + LoRAでのRTX 4090単体での実時間制御達成を目指す

**長期 (6-12ヶ月)**:
1. MTPヘッドをアクションチャンキング専用に再設計（マルチステップアクション同時予測）
2. Gated DeltaNetの再帰状態を「ロボットのワーキングメモリ」として活用する新しいアーキテクチャの提案
3. 実機（マニピュレータ）でのデプロイメントと長時間タスクでの安定性評価

---

## 付録A: config.json完全データ

HuggingFace `Qwen/Qwen3.5-4B` より2026年3月29日に取得:

```json
{
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "image_token_id": 248056,
    "model_type": "qwen3_5",
    "text_config": {
        "attention_bias": false,
        "attention_dropout": 0.0,
        "attn_output_gate": true,
        "dtype": "bfloat16",
        "eos_token_id": 248044,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 2560,
        "initializer_range": 0.02,
        "intermediate_size": 9216,
        "layer_types": [
            "linear_attention","linear_attention","linear_attention","full_attention",
            "linear_attention","linear_attention","linear_attention","full_attention",
            "linear_attention","linear_attention","linear_attention","full_attention",
            "linear_attention","linear_attention","linear_attention","full_attention",
            "linear_attention","linear_attention","linear_attention","full_attention",
            "linear_attention","linear_attention","linear_attention","full_attention",
            "linear_attention","linear_attention","linear_attention","full_attention",
            "linear_attention","linear_attention","linear_attention","full_attention"
        ],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "max_position_embeddings": 262144,
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": false,
        "num_attention_heads": 16,
        "num_hidden_layers": 32,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-06,
        "tie_word_embeddings": true,
        "vocab_size": 248320,
        "mamba_ssm_dtype": "float32",
        "rope_parameters": {
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25
        }
    },
    "vision_config": {
        "depth": 24,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1024,
        "in_channels": 3,
        "intermediate_size": 4096,
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2560,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2
    }
}
```

---

## 付録B: 参考文献・情報源

- [Qwen3.5: Nobody Agrees on Attention Anymore (Maxime Labonne, HuggingFace Blog)](https://huggingface.co/blog/mlabonne/qwen35)
- [Gated Delta Networks: Improving Mamba2 with Delta Rule (arXiv:2412.06464, ICLR 2025)](https://arxiv.org/abs/2412.06464)
- [Gated DeltaNet for Linear Attention (Sebastian Raschka, LLMs-from-scratch)](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/08_deltanet/README.md)
- [Qwen3.5 Gated DeltaNet Analysis (GitHub Gist)](https://gist.github.com/justinchuby/0213aa253664fb72e9adb0089816de15)
- [NVlabs/GatedDeltaNet Official Implementation](https://github.com/NVlabs/GatedDeltaNet)
- [flash-linear-attention (fla-org)](https://github.com/fla-org/flash-linear-attention)
- [StarVLA: Modular VLA Codebase](https://github.com/starVLA/starVLA)
- [Qwen3-VL Vision Encoder Swap Discussion (GitHub Issue #114)](https://github.com/QwenLM/Qwen3-VL/issues/114)
- [vLLM Qwen3.5 Usage Guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
- [Qwen3.5 MTP support in llama.cpp (PR #20700)](https://github.com/ggml-org/llama.cpp/pull/20700)
- [Qwen3.5 Quantization Analysis (kaitchup)](https://kaitchup.substack.com/p/qwen35-quantization-similar-accuracy)
- [INT4 AWQ Quantization Issue (NVIDIA/Model-Optimizer #850)](https://github.com/NVIDIA/Model-Optimizer/issues/850)
- [SpatialVLA-Mamba (OpenReview)](https://openreview.net/forum?id=sTn4EqE49A)
- [RoboMamba (arXiv:2406.04339, NeurIPS 2024)](https://arxiv.org/abs/2406.04339)
- [Qwen3.5 on Unsloth](https://unsloth.ai/docs/models/qwen3.5)
- [Qwen3.5 Fine-Tuning with QLoRA (DataCamp)](https://www.datacamp.com/tutorial/fine-tuning-qwen3-5-small)
- [Qwen 3VL as VLA Model (Anton Maltsev)](https://medium.com/@zlodeibaal/one-of-the-best-vla-models-qwen-3vl-d-551cf9bf2e60)
- [VLA Survey](https://vla-survey.github.io/)
- [Qwen Official Speed Benchmark](https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html)
- [1M Tokens/s: Qwen 3.5 on B200 GPUs](https://medium.com/google-cloud/1-million-tokens-per-second-qwen-3-5-27b-on-gke-with-b200-gpus-161da5c1b592)
