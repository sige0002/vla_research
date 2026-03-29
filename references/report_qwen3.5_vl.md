# Qwen3.5-VL 調査レポート

**作成日**: 2026年3月29日
**対象モデル**: Qwen3.5シリーズ（特にQwen3.5-4B）
**副題**: Towards Native Multimodal Agents

---

## 1. 概要

Qwen3.5は、2026年2月にQwenチーム（Alibaba Cloud）からリリースされたマルチモーダル基盤モデルシリーズである。前世代のQwen3-VLとは異なり、**テキストとビジョンを分離したモデル（Qwen3 + Qwen3-VL）ではなく、統一されたビジョン言語基盤モデル（Unified Vision-Language Foundation Model）** として設計されている点が最大の特徴である。

公式引用:
```bibtex
@misc{qwen3.5,
    title  = {{Qwen3.5}: Towards Native Multimodal Agents},
    author = {{Qwen Team}},
    month  = {February},
    year   = {2026},
    url    = {https://qwen.ai/blog?id=qwen3.5}
}
```

---

## 2. リリース日とモデルサイズ一覧

### 2.1 Denseモデル

| モデル名 | パラメータ数 | リリース日 | pipeline_tag |
|----------|------------|-----------|--------------|
| Qwen3.5-0.8B | 0.8B | 2026-02-28 | image-text-to-text |
| Qwen3.5-2B | 2B | 2026-02-28 | image-text-to-text |
| **Qwen3.5-4B** | **4B** | **2026-02-27** | **image-text-to-text** |
| Qwen3.5-9B | 9B | 2026-02-27 | image-text-to-text |
| Qwen3.5-27B | 27B | 2026-02-24 | image-text-to-text |

### 2.2 MoE（Mixture-of-Experts）モデル

| モデル名 | 総パラメータ/アクティブ | リリース日 |
|----------|----------------------|-----------|
| Qwen3.5-35B-A3B | 35B / 3B active | 2026-02-24 |
| Qwen3.5-122B-A10B | 122B / 10B active | 2026-02-24 |
| Qwen3.5-397B-A17B | 397B / 17B active | 2026-02-16 |

**重要**: Qwen3.5シリーズは全モデルが `image-text-to-text` パイプラインに分類されており、テキスト専用モデルとVLモデルの区別がない。ViTを含む統一アーキテクチャであり、テキストのみで使う場合は `--language-model-only` フラグで無効化可能。

---

## 3. アーキテクチャ詳細

### 3.1 Qwen3-VLからの根本的変更点

Qwen3.5はQwen3-VLとは根本的にアーキテクチャが異なる。以下に主要な変更点を示す。

#### (a) ハイブリッドアテンション: Gated DeltaNet + Gated Attention

Qwen3-VLは標準的なTransformerアテンション（GQA）を使用していたが、Qwen3.5では**Gated DeltaNet（線形アテンション）とGated Attention（フルアテンション）のハイブリッド構成**を採用。

- **レイアウトパターン**: `3 x (Gated DeltaNet -> FFN) -> 1 x (Gated Attention -> FFN)` の繰り返し
- `full_attention_interval: 4` により、4層ごとに1層がフルアテンション
- config.jsonの `layer_types` フィールドに `["linear_attention", "linear_attention", "linear_attention", "full_attention", ...]` として明記

**Gated DeltaNetの詳細（4Bモデル）:**
- Linear Attention Heads（V）: 32
- Linear Attention Heads（QK）: 16
- Head Dimension: 128
- Conv Kernel Dim: 4

**Gated Attentionの詳細（4Bモデル）:**
- Attention Heads（Q）: 16
- KV Heads: 4
- Head Dimension: 256
- Partial Rotary Factor: 0.25（RoPEの25%のみ回転）

この設計により、長コンテキスト処理時の計算効率が大幅に向上している。線形アテンションは O(n) の計算量であり、262Kトークンのネイティブコンテキスト長をサポートしつつ、YaRNスケーリングで最大1,010,000トークンまで拡張可能。

#### (b) Qwen3-VL vs Qwen3.5 テキストモデル比較（4Bクラス）

| パラメータ | Qwen3-VL-4B | Qwen3.5-4B |
|-----------|-------------|------------|
| アーキテクチャ | Qwen3VLForConditionalGeneration | Qwen3_5ForConditionalGeneration |
| hidden_size | 2560 | 2560 |
| num_hidden_layers | 36 | 32 |
| num_attention_heads | 32 | 16（full） / 16（linear QK） |
| num_key_value_heads | 8 | 4（full） |
| head_dim | 128 | 256（full） / 128（linear） |
| intermediate_size | 9728 | 9216 |
| vocab_size | 151,936 | 248,320 |
| max_position_embeddings | 262,144 | 262,144 |
| アテンション方式 | 全層フルアテンション | ハイブリッド（3:1 linear:full） |
| MTP | なし | あり（mtp_num_hidden_layers: 1） |
| mamba_ssm_dtype | なし | float32 |
| attn_output_gate | なし | あり |

#### (c) Vision Encoder (ViT) の変更点

| パラメータ | Qwen2.5-VL-3B | Qwen3-VL-4B | Qwen3.5-4B |
|-----------|---------------|-------------|------------|
| **patch_size** | **14** | **16** | **16** |
| hidden_size | 1280 | 1024 | 1024 |
| depth | 32 | 24 | 24 |
| num_heads | 16 | 16 | 16 |
| intermediate_size | 3420 | 4096 | 4096 |
| hidden_act | silu | gelu_pytorch_tanh | gelu_pytorch_tanh |
| spatial_merge_size | 2 | 2 | 2 |
| temporal_patch_size | 2 | 2 | 2 |
| window_size | 112 | なし | なし |
| fullatt_block_indexes | [7,15,23,31] | なし | なし |
| **deepstack_visual_indexes** | **なし** | **[5, 11, 17]** | **[]（空）** |
| out_hidden_size | 2048 | 2560 | 2560 |
| num_position_embeddings | なし | 2304 | 2304 |

**重要な発見: DeepStackの廃止**

Qwen3-VLではViTの中間層出力をLLMに注入する `deepstack_visual_indexes: [5, 11, 17]` が採用されていたが、**Qwen3.5では全モデルで `deepstack_visual_indexes: []` と空配列になっており、DeepStackが廃止されている**。これは大きなアーキテクチャ変更である。

### 3.2 モデルサイズ別ViT構成

| モデル | ViT depth | ViT hidden_size | ViT num_heads | ViT intermediate_size | patch_size |
|--------|----------|-----------------|---------------|----------------------|------------|
| 0.8B | 12 | 768 | 12 | 3072 | 16 |
| 2B | 24 | 1024 | 16 | 4096 | 16 |
| **4B** | **24** | **1024** | **16** | **4096** | **16** |
| 9B | 27 | 1152 | 16 | 4304 | 16 |
| 27B | 27 | 1152 | 16 | 4304 | 16 |
| 35B-A3B (MoE) | 27 | 1152 | 16 | 4304 | 16 |
| 122B-A10B (MoE) | 27 | 1152 | 16 | 4304 | 16 |

### 3.3 モデルサイズ別LLM構成

| モデル | hidden_size | layers | attn_heads(full) | kv_heads | intermediate | linear_V_heads | linear_QK_heads |
|--------|-----------|--------|-----------------|----------|-------------|----------------|-----------------|
| 0.8B | 1024 | 24 | 8 | 2 | 3584 | 16 | 16 |
| 2B | 2048 | 24 | 8 | 2 | 6144 | 16 | 16 |
| **4B** | **2560** | **32** | **16** | **4** | **9216** | **32** | **16** |
| 9B | 4096 | 32 | 16 | 4 | 12288 | 32 | 16 |
| 27B | 5120 | 64 | 24 | 4 | 17408 | 48 | 16 |

### 3.4 MoEモデル構成

| パラメータ | 35B-A3B | 122B-A10B | 397B-A17B |
|-----------|---------|-----------|-----------|
| hidden_size | 2048 | 3072 | 4096 |
| num_hidden_layers | 40 | 48 | 64（推定） |
| num_experts | 256 | 256 | 256（推定） |
| num_experts_per_tok | 8 | 8 | 8（推定） |
| moe_intermediate_size | 512 | 1024 | （確認中） |
| shared_expert_intermediate_size | 512 | 1024 | （確認中） |

### 3.5 新規アーキテクチャ要素まとめ

1. **Gated DeltaNet（線形アテンション）**: 計算量O(n)で長コンテキスト処理を効率化
2. **ハイブリッド構成**: 4層ごとに1層のフルアテンション（3:1比率）
3. **attn_output_gate**: アテンション出力にゲーティング機構追加
4. **Multi-Token Prediction (MTP)**: 推論速度向上のための投機的デコーディング対応
5. **DeepStack廃止**: ViT中間層のLLM注入メカニズムを全モデルで無効化
6. **拡大されたVocabulary**: 151,936 -> 248,320トークン
7. **Partial Rotary Factor**: RoPEの25%のみを回転埋め込みに使用
8. **mamba_ssm_dtype: float32**: Mamba SSMの計算精度指定（内部でSSM的処理の可能性）

---

## 4. ベンチマーク性能

### 4.1 Vision-Language ベンチマーク（Qwen3.5-4B）

Qwen3.5-4Bは、**前世代のQwen3-VL-30B（約30Bパラメータ）を多くのベンチマークで上回る**という驚異的な性能を示している。

| カテゴリ | ベンチマーク | Qwen3.5-4B | Qwen3-VL-30B | 差分 |
|---------|------------|-----------|-------------|------|
| STEM | MMMU | 77.6 | -- | -- |
| STEM | MMMU-Pro | 66.3 | -- | -- |
| STEM | MathVision | **74.6** | 65.7 | **+8.9** |
| STEM | MathVista(mini) | **85.1** | -- | -- |
| General VQA | RealWorldQA | **79.5** | 77.4 | **+2.1** |
| General VQA | MMStar | **78.3** | 75.5 | **+2.8** |
| General VQA | MMBenchEN | **89.4** | 88.9 | **+0.5** |
| OCR | OmniDocBench1.5 | **86.2** | -- | -- |
| OCR | CharXiv(RQ) | **70.8** | 56.6 | **+14.2** |
| Spatial | CountBench | **96.3** | 90.0 | **+6.3** |
| Video | VideoMME(w sub.) | **83.5** | 79.9 | **+3.6** |
| Video | VideoMME(w/o sub.) | **76.9** | 73.3 | **+3.6** |
| Video | MLVU | **82.8** | 78.9 | **+3.9** |

### 4.2 Qwen3.5-27Bのベンチマーク

| ベンチマーク | Qwen3.5-27B |
|------------|-----------|
| MMMU | 82.3 |
| MMMU-Pro | 75.0 |
| MathVision | 86.0 |
| VideoMME(w sub.) | 87.0 |
| VideoMMMU | 82.3 |
| MLVU | 85.9 |
| SWE-bench Verified | 72.4 |
| LiveCodeBench v6 | 80.7 |
| HMMT Feb 25 | 92.0 |

### 4.3 言語ベンチマーク（Qwen3.5-4B）

| ベンチマーク | Qwen3.5-4B |
|------------|-----------|
| MMLU-Pro | 79.1 |
| C-Eval | 85.1 |
| IFEval | 89.8 |
| HMMT Feb 25 | 74.0 |
| LiveCodeBench v6 | 55.8 |
| MMMLU（多言語） | 76.1 |

---

## 5. ビデオ理解能力

Qwen3.5は前世代と比較して大幅なビデオ理解性能の向上を示している。

### 5.1 ビデオ処理仕様
- デフォルトFPS: 2
- フレームサンプリング: `do_sample_frames: True`
- 長時間ビデオ（1時間規模）対応: `longest_edge=469,762,048`（約224Kビデオトークン）
- 262Kコンテキスト長により、長時間ビデオのフレーム系列を直接処理可能

### 5.2 ビデオベンチマーク

| ベンチマーク | Qwen3.5-4B | Qwen3.5-27B |
|------------|-----------|-------------|
| VideoMME(w sub.) | 83.5 | 87.0 |
| VideoMME(w/o sub.) | 76.9 | -- |
| VideoMMMU | 74.1 | 82.3 |
| MLVU | 82.8 | 85.9 |

---

## 6. ロボティクス応用・VLA関連

### 6.1 モデルカードでの言及

**Qwen3.5のモデルカードおよびブログには、ロボティクス、VLA（Vision-Language-Action）、embodied AIに関する直接的な言及は確認されなかった。**

### 6.2 VLA統合に関する考察

ただし、以下のアーキテクチャ特性はVLA統合に直接的な利点を持つ：

1. **統一されたVision-Languageモデル**: テキストとビジョンの早期融合（early fusion）により、視覚的理解と言語的推論が密結合。VLAのバックボーンとして、別々のモデルを組み合わせる必要がない。

2. **Gated DeltaNet（線形アテンション）**: 計算量O(n)の線形アテンションにより、リアルタイムロボット制御に必要な低遅延推論が可能。**ロボット制御ループでの使用において、フルアテンションの O(n^2) と比較して大きな利点。**

3. **DeepStackの廃止**: ViTの中間層出力をLLMに注入する仕組みが廃止されたことは、ViT部分を独立して扱いやすくなったことを意味する。VLAでViTエンコーダを共有・切り離す際のクリーンなインターフェースを提供。

4. **patch_size=16**: Qwen2.5-VLの14から16に変更。パッチ数が減少するため、画像あたりのトークン数が削減され、推論速度が向上。ロボット制御のリアルタイム要件に有利。

5. **262Kコンテキスト長**: 長いタスク軌跡やデモンストレーション系列を入力可能。

6. **MTP（Multi-Token Prediction）**: 投機的デコーディングにより推論速度向上。ロボット制御での応答遅延を低減。

7. **4Bモデルの存在**: エッジデバイス・ロボットに搭載可能なサイズでありながら、Qwen3-VL-30Bを上回る性能。**研究提案で言及されている「qwen3.5 4B」はVLA統合の有力候補。**

### 6.3 VLA統合時の技術的考慮事項

- **ViTエンコーダ出力**: `out_hidden_size` がLLMの `hidden_size` と一致（4Bモデルでは2560）。アクションヘッドの追加が容易。
- **spatial_merge_size=2**: 2x2のパッチマージにより、空間解像度とトークン数のトレードオフを制御。
- **temporal_patch_size=2**: ビデオの時間方向で2フレームを1パッチに。ロボットの動作予測に利用可能。
- **Gated DeltaNetのリカレント特性**: 線形アテンションはRNNに近い特性を持ち、逐次的なトークン生成（アクション生成）に適している。

---

## 7. Qwen2.5-VL / Qwen3-VL / Qwen3.5 アーキテクチャ進化まとめ

| 特性 | Qwen2.5-VL | Qwen3-VL | Qwen3.5 |
|------|-----------|----------|---------|
| リリース | 2024年末-2025年初 | 2025年9-10月 | 2026年2月 |
| モデル分離 | VL専用 | VL専用 | **統一モデル** |
| ViT patch_size | 14 | 16 | 16 |
| ViT hidden_act | silu | gelu_pytorch_tanh | gelu_pytorch_tanh |
| ViT window attention | あり | なし | なし |
| **DeepStack** | **なし** | **あり** | **廃止（空配列）** |
| アテンション方式 | Full Attention (GQA) | Full Attention (GQA) | **Hybrid (DeltaNet + Full)** |
| コンテキスト長 | 128K | 262K | 262K (最大1M+) |
| vocab_size | 151,936 | 151,936 | **248,320** |
| MTP | なし | なし | **あり** |
| 言語サポート | 限定的 | 拡張 | **201言語** |
| Thinkingモード | なし | Thinking版のみ | **デフォルト搭載** |
| MoEモデル | なし | 30B-A3B | 35B-A3B, 122B-A10B, 397B-A17B |

---

## 8. 技術報告書・論文

### 8.1 公式ブログ
- URL: https://qwen.ai/blog?id=qwen3.5
- タイトル: "Qwen3.5: Towards Native Multimodal Agents"

### 8.2 arxiv論文
本調査時点（2026年3月29日）では、Qwen3.5の独立した技術報告書（arxivプレプリント）は確認できなかった。ただし、HuggingFaceモデルカードに詳細なアーキテクチャ情報とベンチマーク結果が記載されている。

---

## 9. 研究提案への示唆

### 9.1 「Qwen3.5-4B」をVLAターゲットVLMとする妥当性

**極めて妥当である。** 以下の理由による：

1. **性能対コスト比**: 4Bパラメータで前世代の30Bクラスを上回る視覚理解性能。ロボットのエッジデバイスでの動作が現実的。

2. **ハイブリッドアテンション**: Gated DeltaNetの線形アテンションにより、長いアクション系列の処理が計算量的に実現可能。フルアテンションのみの場合と比較して、特にコンテキスト長が長い場合（デモンストレーション動画の処理等）で大きな利点。

3. **統一アーキテクチャ**: ViTとLLMが最初から統合されているため、VLAのファインチューニング時にEnd-to-Endで学習可能。

4. **DeepStack廃止によるクリーンなViT出力**: ViTの最終層出力のみがLLMに渡されるシンプルな設計。アクションヘッドの接続ポイントが明確。

5. **MTP対応**: 投機的デコーディングによる推論速度向上がロボット制御の応答時間要件を満たす助けになる。

### 9.2 注意点

- **DeepStack廃止の影響**: Qwen3-VLではDeepStackにより視覚的特徴の多スケール融合が行われていたが、Qwen3.5ではこれが無くなっている。VLA用途で中間層視覚特徴が重要な場合、独自にDeepStack的機構を追加する必要がある可能性。
- **Gated DeltaNetの学習安定性**: VLAファインチューニング時にGated DeltaNetの線形アテンション層がどの程度安定して学習できるかは未検証。
- **ロボティクス専用のベンチマーク**: Qwen3.5のモデルカードにはロボティクス固有のベンチマーク（例: CALVIN, Language-Table等）の結果は報告されていない。

---

## 10. 結論

Qwen3.5は2026年2月にリリースされた次世代の統一ビジョン言語モデルであり、**Gated DeltaNetによるハイブリッドアテンション**、**DeepStackの廃止**、**Multi-Token Prediction**、**拡張されたボキャブラリ（248K）** 等の大幅なアーキテクチャ刷新が行われている。

特に**Qwen3.5-4B**は、4Bパラメータという軽量さで前世代の30Bクラスモデルに匹敵する視覚理解性能を達成しており、ロボティクスVLAのバックボーンVLMとして非常に有望である。線形アテンションの採用による低遅延推論、統一モデルとしてのEnd-to-End学習の容易さ、262Kコンテキスト長による長時間ビデオ/軌跡処理能力は、VLA統合において大きな技術的優位性を提供する。

ただし、ロボティクス分野での直接的な評価結果は公開されておらず、VLAファインチューニング時のGated DeltaNetの挙動検証やDeepStack廃止の影響評価は独自に行う必要がある。
