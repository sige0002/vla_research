# Qwen3.5-4B VLAバックボーン技術書

> 本書はQwen3.5-4Bを軸としたVLA構築の技術的基盤をまとめる。
> ViT置換（C-RADIO等）は選択肢の一つとして記載するが、採用は未定。

---

## 1. なぜQwen3.5-4Bか

| 評価軸 | Qwen3.5-4B | Qwen3-VL-4B | Gemma 3-4B | PaliGemma-3B |
|--------|-----------|-------------|------------|--------------|
| VLA実績 | StarVLA統合済 | Xiaomi-R0 (98.7%) | pi0.6 | pi0/pi0.5 |
| アテンション | **ハイブリッド (75%線形)** | 全層Full | 全層Full | 全層Full |
| 推論計算量 | **O(n)優位** | O(n^2) | O(n^2) | O(n^2) |
| DeepStack | **廃止（クリーン出力）** | あり [5,11,17] | なし | なし |
| ERQA (具身推論) | **67.5** | 52.5 | - | - |
| CountBench | **96.3** | 90.0 | - | - |
| コンテキスト | 262K (max 1M+) | 262K | 128K | 8K |
| MTP | **あり** | なし | なし | なし |
| 動画理解 | VideoMME 83.5 | 79.9 | - | - |

**選定理由**:
1. 4Bで前世代30Bクラスを超える性能
2. Gated DeltaNet（線形アテンション）で長時間制御が効率的
3. DeepStack廃止でViT出力がシンプル → アーキテクチャ拡張が容易
4. 具身推論（ERQA）で+15pt → ロボット用途に直結する能力
5. StarVLAが既にQwen3.5-4B統合済み → 実装基盤あり

---

## 2. 時系列処理: MEM的改造は必要か

> **結論: 必要。Qwen3.5のViTにはフレーム間attentionが存在しない。MEM的改造なしではVLAの時系列処理は非効率。**

### 2.0.1 Qwen3.5の動画処理の実態

Qwen開発チーム自身の回答（GitHub Issue #753）:
> "The only interactive attention along time dimension at vision tower is the 3D Conv of patch embed layer"

つまり:
1. **3D Conv（temporal_patch_size=2）** で隣接2フレームを局所結合 → これが**唯一**の時間融合
2. ViT内の24層は**フレームを完全に独立に処理**（フレーム間attentionなし）
3. **全フレームの全トークンがそのままLLMに渡される**（過去フレームドロップなし）

### 2.0.2 トークン数の爆発

| フレーム数 | Qwen3.5（全トークンLLMへ） | MEM（現在フレームのみ） |
|-----------|--------------------------|----------------------|
| 1画像 | 64トークン | 64トークン |
| 10フレーム | **320トークン** | 64トークン |
| 50フレーム (5Hz×10秒) | **1,225トークン** | 64トークン |
| 100フレーム | **3,200トークン** | 64トークン |

VLAで5Hz×10秒の文脈を入れると、Qwen3.5はMEMの**約20倍のトークン**をLLMに渡す。

### 2.0.3 MEMとの設計思想の根本的差異

```
Qwen3.5:
  Frame1 ──► ViT(独立) ──► 全トークン ─┐
  Frame2 ──► ViT(独立) ──► 全トークン ─┤──► LLM（大量トークン）
  Frame3 ──► ViT(独立) ──► 全トークン ─┘

MEM:
  Frame1 ──┐
  Frame2 ──┤──► ViT(時間attention) ──► 現在フレームのみ ──► LLM（少量トークン）
  Frame3 ──┘     ↑4層ごとにcausal temporal attn
                  ↑上位層で過去フレームドロップ
```

### 2.0.4 必要な改造

Qwen3.5のViT（24層）に以下を追加する:

1. **Causal Temporal Attention**: 6層ごと（layer 5, 11, 17, 23）に、同一空間位置パッチの時間方向因果的attention
2. **過去フレームドロップ**: layer 18以降で現在フレーム以外のトークンを破棄
3. **正弦波時間PE**: e(0)=0で単一画像互換性を維持（追加パラメータなし）

Qwen3.5はDeepStack廃止済みなので、MEMのattentionパターン変更だけで済む（Qwen3-VLだとDeepStackとの整合が必要で複雑になる）。**DeepStack廃止はMEM改造にとってプラス。**

---

## 3. StarVLAによるVLA化の実装手順

StarVLA（https://github.com/starVLA/starVLA）がQwen3.5-4Bを統合済み。VLA化の具体的ステップ:

### 3.1 全体フロー

```
Step 1: モデル準備 (Qwen3.5-4B-Instruct DL)
Step 2: [FAST使用時のみ] 語彙拡張 (アクション専用トークン追加)
Step 3: YAML設定作成 (フレームワーク・データ・学習率指定)
Step 4: 訓練実行 (DeepSpeed ZeRO-2, 差分学習率)
Step 5: 評価 (LIBERO/SimplerEnv)
```

### 3.2 アクションヘッド5方式の比較

| 方式 | 生成方法 | 速度 | 精度 | MTP親和性 | 推奨度 |
|------|---------|------|------|----------|--------|
| **QwenFAST** | 離散トークン自己回帰 | 中 | 中-高 | **高** | 初手 |
| **QwenOFT** | MLP並列デコード | **最速** | 中 | 低 | シンプル開始 |
| **QwenGR00T** | Flow Matching DiT | 中 | 高 | 低 | 高精度 |
| **QwenPI** | 層別Flow Matching | 中 | **最高** | 低 | 最終形 |
| **QwenAdapter** | 学習可能クエリ+多層特徴 | 中 | 高 | 中 | 高度 |

### 3.3 VLM→VLA変換の本質

StarVLAのコード構造から明らかになったVLM→VLA変換の3つの核心:

1. **隠れ状態→アクション空間マッピング**: VLMの `[B, seq, 2560]` 出力をアクションヘッドで `[B, horizon, action_dim]` に変換
2. **固有受容覚の注入**: 関節角度等をMLP射影してアクションヘッドに入力（QwenGR00T/PI: state_encoder、QwenAdapter: ProprioProjector）
3. **共同訓練**: VLA損失 + VLM損失（0.1倍スケール）で言語理解能力を維持

### 3.4 学習率戦略

```yaml
learning_rate:
  qwen_vl_interface: 1.0e-5   # VLMバックボーン（低LR、事前学習保護）
  action_model: 1.0e-4         # アクションヘッド（高LR、高速収束）
```

### 3.5 主要ファイルパス

| 用途 | パス |
|------|------|
| Qwen3.5ラッパー | `starVLA/model/modules/vlm/QWen3_5.py` |
| フレームワーク | `starVLA/model/framework/Qwen{FAST,OFT,GR00T,PI,Adapter}.py` |
| アクションヘッド | `starVLA/model/modules/action_model/` |
| 訓練 | `starVLA/training/train_starvla_cotrain.py` |
| 語彙拡張 | `starVLA/model/modules/vlm/tools/add_qwen_special_tokens/` |

---

## 4. アーキテクチャ詳細

### 2.1 全体構成

```
Image/Video ──► ViT (24層, 1024dim, patch16) ──► Spatial Merge (2x2)
                                                       │
                                                  Projector (→2560dim)
                                                       │
Text ──────────────────────────────────────────────────┤
                                                       ▼
                                              LLM (32層, 2560dim)
                                              ┌─ linear_attention × 3
                                              ├─ full_attention   × 1
                                              └─ ×8ブロック = 32層
                                                       │
                                                  Token Output
```

### 2.2 LLM部（Gated DeltaNet + Full Attention）

**レイヤー構成** (32層):
```
[linear, linear, linear, full] × 8ブロック
→ 24層 Gated DeltaNet (75%) + 8層 Full Attention (25%)
```

| パラメータ | Full Attention層 | Linear Attention層 |
|-----------|-----------------|-------------------|
| Q heads | 16 | - |
| KV heads | 4 (GQA) | K:16, V:32 |
| head_dim | 256 | K:128, V:128 |
| 位置符号化 | RoPE (partial 25%) | Conv kernel=4 |
| 計算量 | O(n^2) | **O(n)** |
| KVキャッシュ | 系列長に比例 | **固定サイズ** |

**Gated DeltaNetの動作原理**（ICLR 2025採択）:
```
1. 状態減衰:  S_t = g_t * S_{t-1}           # 適応的忘却
2. 予測:      pred_t = (S_t * k_t).sum()     # 記憶からの予測
3. 誤り訂正:  Δ_t = (v_t - pred_t) * β_t    # デルタルール
4. 状態更新:  S_t = S_t + k_t * Δ_t          # 差分のみ更新
5. 出力:      y_t = (S_t * q_t).sum() * gate  # ゲート付き出力
```

VLAにとっての本質的な利点:
- **固定サイズ再帰状態**: 長時間タスク（15分+）でもメモリ・速度が一定
- **誤り訂正的更新**: 新しい観測で古い情報を適応的に修正 → ロボットの状況変化追従に適合
- **デコード時O(1)**: アクショントークン生成が系列長に依存しない

### 2.3 ViT部

| パラメータ | 値 |
|-----------|-----|
| depth | 24 |
| hidden_size | 1024 |
| num_heads | 16 |
| patch_size | 16 |
| temporal_patch_size | 2 |
| spatial_merge_size | 2 |
| out_hidden_size | 2560 (= LLM hidden) |
| deepstack | **[]（廃止）** |

- ViT最終層出力のみがProjector経由でLLMに渡される（中間層注入なし）
- temporal_patch_size=2: 動画の連続2フレームを1パッチ化（時間的圧縮）
- spatial_merge_size=2: 2x2パッチ統合で視覚トークン数を1/4に削減

### 2.4 MTP（Multi-Token Prediction）

- 1層Transformerブロックによるnext-next token予測
- 推論時: speculative decodingで1.5-2倍のスループット向上
- vLLM対応: `--speculative-config '{"method": "mtp", "num_speculative_tokens": N}'`
- FastMTP最適化: ドラフト生成 22ms → 6ms (3.7倍高速)

**VLAでの活用可能性**:
- アクションチャンキング（複数トークン同時予測）と自然に整合
- ただしアクショントークンの分布はテキストと異なるため、アクセプタンスレート低下の懸念あり

---

## 3. 推論性能

### 3.1 速度見積もり

| 構成 | 推定レイテンシ | 推定制御周波数 |
|------|-------------|--------------|
| BF16, RTX 4090 | ~150-200 tok/s | - |
| 7dim action × 1step | ~47ms | ~21Hz |
| 7dim action × 10step chunk | ~470ms → 5Hz推論, チャンク再利用で30-100Hz | ○ |
| MTP有効化時 | 上記×0.5-0.7 | さらに向上 |

### 3.2 メモリ効率

- BF16パラメータ: ~8GB
- KVキャッシュ: **8層分のみ成長**（24層は固定サイズ状態）
- → 全層Fullの場合比で**約75%のKVキャッシュ削減**
- RTX 4090 (24GB): BF16で十分動作可能

### 3.3 長時間タスクでの安定性

```
Fullアテンションモデル:  推論速度 ──時間→ 漸次低下（KVキャッシュ肥大）
Qwen3.5 (75%線形):      推論速度 ──時間→ ほぼ一定（固定状態）
```

15分級のロングホライズンタスクで、推論速度の安定性は実用上の大きな利点。

---

## 4. VLA統合の実績と手法

### 4.1 StarVLA（既存フレームワーク）

StarVLAがQwen3.5-4Bを統合済み。4つのアクション生成方式を提供:

| 方式 | 生成方法 | 速度 | 精度 | MTP親和性 |
|------|---------|------|------|----------|
| **Qwen-FAST** | 離散トークン自己回帰 | 中 | 中 | **高** |
| **Qwen-OFT** | MLP並列デコード | **高** | 中 | 低 |
| **Qwen-PI** | Flow Matching | 低 | **高** | 低 |
| **Qwen-GR00T** | VL推論+高速予測 | 高 | 高 | 中 |

### 4.2 Xiaomi-R0の知見（Qwen3-VLベース）

Xiaomi-R0はQwen3-VL-4B + DiT Flow Matchingで:
- LIBERO 98.7%, 推論80ms, 30Hz
- Lambda-Shape注意マスク、RoPE位置オフセット、動的損失再重み付け

これらの技術はQwen3.5にも適用可能。DeepStack廃止により、むしろアクションヘッド設計がシンプルになる。

---

## 5. ファインチューニング戦略

### 5.1 LoRA/QLoRA

- Unsloth, StarVLAがQwen3.5 LoRAをサポート済み
- **推奨**: BF16 LoRA（4Bモデルはメモリ的に全精度LoRAが可能）
- 線形アテンション層とFull層でLoRAランクを変える検討が必要

### 5.2 量子化

| 方式 | 推奨度 | 注意点 |
|------|--------|--------|
| FP8 | ○ | H100のFP8テンソルコア活用可 |
| INT4 (AWQ/GPTQ) | △ | **アテンション層は非量子化にすべき** |
| 3bit以下 | × | Qwen系は低ビットで性能劣化が顕著 |

### 5.3 VLAファインチューニング時の注意点

1. `mamba_ssm_dtype: float32` → SSM関連計算はfloat32精度で実行すべき
2. ViT交換はQwen公式が**非推奨**（動的解像度アライメントに多大な工数）
3. アクションヘッドはfull precisionで学習
4. MTPヘッドのVLAファインチューニング安定性は未検証 → 初期はMTPなしで学習推奨

---

## 6. ViT拡張の選択肢（採用未定）

ViT交換はQwen公式が非推奨としているが、研究要素として検討する価値はある。以下に選択肢を整理する。

### 6.1 選択肢A: Qwen3.5 ViTをそのまま使用（推奨）

- 最もリスクが低い
- StarVLA/Xiaomi-R0の実績が直接参照可能
- 動的解像度、temporal_patch_size等の機能が保持される
- **研究新規性**: Gated DeltaNet LLMとの組み合わせ自体が新規

### 6.2 選択肢B: C-RADIO置換

| 項目 | 詳細 |
|------|------|
| C-RADIO出力dim | 1152 |
| Qwen3.5 LLM入力dim | 2560 |
| パッチサイズ | 16（一致） |
| DeepStack | 不要（Qwen3.5で廃止済み） |
| Projector | 2層MLP (1152 → GELU → 2560) + Spatial Merge |

利点: SigLIP2+DINOv3+SAM3の統合特徴、空間特徴エントロピーの高さ
リスク: 動的解像度喪失、temporal_patch_size非対応、大規模再学習が必要

### 6.3 選択肢C: アダプタ追加（折衷案）

- Qwen3.5 ViTは維持
- C-RADIOやDINOv2の特徴を**追加入力**として統合
- 例: DeepVision-VLAのVL-MoT方式で深層に視覚特徴を注入
- リスク最小で追加的な視覚能力を獲得可能

### 6.4 選択肢D: 3D表現の追加

- SpatialVLAのEgo3D Position Encoding
- Any3D-VLAの多様点群統合
- 深度推定アダプタの追加
- 空間理解の強化に有効、ViT本体の変更不要

---

## 7. アーキテクチャ進化の系譜

```
Qwen2.5-VL-3B (2024末)
  ViT: depth=32, dim=1280, patch=14, Window+Full Attention
  LLM: dim=2048, 36層, Full Attention, 128K context
  特徴: 大きいViT、シンプルなLLM
       │
       ▼
Qwen3-VL-4B (2025.9)
  ViT: depth=24, dim=1024, patch=16 (小型化)
  LLM: dim=2560, 36層, Full Attention, 262K context
  特徴: DeepStack [5,11,17] でViT小型化を補償
       │
       ▼
Qwen3.5-4B (2026.2) ← 採用
  ViT: depth=24, dim=1024, patch=16 (同等)
  LLM: dim=2560, 32層, Hybrid (75%線形+25%Full), 262K context
  特徴: DeepStack廃止、Gated DeltaNet導入、MTP追加
        統一マルチモーダルモデル（VL分離なし）
```

進化の方向性: ViTは小型化・安定化、LLMは効率化（線形アテンション）へ。VLAバックボーンとしてはQwen3.5の設計が最も合理的。

---

## 8. Gated DeltaNetとMEM（メモリ機構）の親和性

Qwen3.5のGated DeltaNetは、MEM（Multi-Scale Embodied Memory）との統合において独自の利点を持つ。

### 8.1 再帰状態 = 暗黙的ワーキングメモリ

DeltaNetの再帰状態 `S_t` は固定サイズの連想記憶として機能する。これは:
- MEM短期メモリ（ビデオ特徴の圧縮保持）の一部を**LLM内部で暗黙的に実現**する可能性
- フルアテンションでは全過去トークンを保持するが、DeltaNetでは「重要な情報のみ誤り訂正的に保持」
- → ロボットの状況追跡に適した選択的記憶

### 8.2 長期メモリとの相補性

- DeltaNetの再帰状態: 短期的な文脈保持（数秒〜数十秒）
- MEM長期言語メモリ: 明示的なテキスト記録（数分〜数十分）
- → 二重のメモリ階層が自然に構成される

### 8.3 8層Full Attentionの役割

4層ごとに挿入されるFull Attention層が、DeltaNetの再帰状態では捕捉しきれない**長距離依存関係**を補完する。VLAでは:
- タスク指示と現在の観測の対応付け
- 長期メモリテキストとの照合
- 視覚トークンとの精密なクロスモーダルアテンション

---

## 参考文献

### Qwen3.5
- [Qwen3.5: Towards Native Multimodal Agents](https://qwen.ai/blog?id=qwen3.5)
- [Qwen3.5-4B config.json (HuggingFace)](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Qwen3.5: Nobody Agrees on Attention Anymore](https://huggingface.co/blog/mlabonne/qwen35)
- [Qwen3-VL ViT Swap Discussion (Issue #114)](https://github.com/QwenLM/Qwen3-VL/issues/114)

### Gated DeltaNet
- [Gated Delta Networks (arXiv:2412.06464, ICLR 2025)](https://arxiv.org/abs/2412.06464)
- [NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet)
- [flash-linear-attention (fla-org)](https://github.com/fla-org/flash-linear-attention)

### VLA統合
- [StarVLA](https://github.com/starVLA/starVLA)
- [Xiaomi-Robotics-0 (arXiv:2602.12684)](https://arxiv.org/abs/2602.12684)
- [RoboMamba (NeurIPS 2024)](https://arxiv.org/abs/2406.04339)
- [SpatialVLA-Mamba (OpenReview)](https://openreview.net/forum?id=sTn4EqE49A)

### ViT/視覚バックボーン
- [C-RADIOv4 (arXiv:2601.17237)](https://arxiv.org/abs/2601.17237)
- [Theia: Distilling Diverse VFMs (arXiv:2407.20179)](https://arxiv.org/abs/2407.20179)
- [DeepVision-VLA (arXiv:2603.15618)](https://arxiv.org/abs/2603.15618)
