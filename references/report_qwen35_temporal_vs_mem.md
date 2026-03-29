# Qwen3.5 (Qwen3-VL) の時系列処理とMEMアーキテクチャの比較 -- VLA応用に向けた技術調査レポート

## 1. エグゼクティブサマリー

**結論: Qwen3-VLのネイティブ動画処理では、MEM的な効率的時系列情報圧縮はできない。VLAに適用するにはMEM的な改造が必要である。**

その理由は以下の3点に集約される:

1. **Qwen3-VLのViTにはフレーム間attention（時間的attention）が存在しない** -- 3D Conv以外にフレーム間の情報交換メカニズムがない
2. **全フレームのトークンがそのままLLMに渡される** -- 過去フレームのドロップや時系列圧縮が一切ない
3. **トークン数がフレーム数に比例して線形増大する** -- VLAの実時間制御には致命的

---

## 2. Qwen3-VL / Qwen2.5-VLの動画処理アーキテクチャ詳細

### 2.1 パッチ埋め込み: 3D Convolutionによる時空間パッチ化

Qwen2.5-VLおよびQwen3-VLは、`nn.Conv3d`を用いた3次元畳み込みでビデオフレームをパッチ化する。

```python
# Qwen2.5-VL / Qwen3-VL共通
kernel_size = [temporal_patch_size, patch_size, patch_size]  # [2, 14, 14] or [2, 16, 16]
self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)
```

**動作原理:**
- `temporal_patch_size=2`: 連続する2フレームを1つの時間パッチとして結合
- `patch_size=14`（Qwen2.5-VL）/ `16`（Qwen3-VL）: 空間方向の14x14 or 16x16ピクセルを1パッチに
- 入力形状: `(batch, 3, 2, 14, 14)` → 出力形状: `(batch, embed_dim, 1, 1, 1)` → flatten → `(N, embed_dim)`

**重要:** この3D Convは**唯一の時間方向の情報融合メカニズム**である。Qwen開発チームのメンテナが明言:

> "The only interactive attention happens along time dimension at vision tower is the 3D Conv of patch embed layer"
> (GitHub Issue #753, QwenLM/Qwen3-VL)

つまり、隣接する2フレームは3D Convで局所的に融合されるが、それ以降のViT層では**フレーム間のattentionは一切行われない**。

### 2.2 ViTのAttention構造: フレーム間attentionなし

#### Qwen2.5-VL
- **32層**のVision Transformer
- 大部分の層: **Window Attention**（112x112ピクセル = 8x8パッチのウィンドウ内）
- 4層のみ（layer 7, 15, 23, 31）: **Full Attention**（全パッチに対するattention）
- **cu_seqlens**により、各画像/動画フレームのattentionは完全に分離
- Full Attention層でも**フレーム間のattentionは発生しない**（attentionマスクが-infでブロック）

#### Qwen3-VL
- **27層**のVision Transformer
- Window Attention + 一部Full Attention（Qwen2.5-VLと同様の構造）
- **DeepStack**: layer 8, 16, 24の中間特徴をLLMの対応するデコーダ層に注入
- やはり**フレーム間のattentionは存在しない**

Qwen開発チームの公式回答:

> "We encode/tokenize each image separately via the vision transformer, without any attention interaction between images. The interaction between images will be implemented in the LLM blocks."

### 2.3 Spatial Merge（空間マージ）

ViT出力後、`PatchMerger`が空間方向の圧縮を行う:

```python
# spatial_merge_size = 2 → 2x2パッチを1トークンに統合
self.hidden_size = context_dim * (spatial_merge_size**2)  # 4パッチ分を結合
self.mlp = nn.Sequential(
    nn.Linear(self.hidden_size, self.hidden_size),
    nn.GELU(),
    nn.Linear(self.hidden_size, dim),
)
```

これは純粋に**空間方向の圧縮**であり、時間方向の圧縮は行わない。

### 2.4 トークン数の計算式

LLMに渡されるビジュアルトークン数:

```
tokens = (grid_t × grid_h × grid_w) / (spatial_merge_size²)
```

ここで:
- `grid_t = num_frames / temporal_patch_size` （temporal_patch_size=2なので、フレーム数/2）
- `grid_h = height / patch_size`
- `grid_w = width / patch_size`
- `spatial_merge_size = 2` → 4分の1に空間圧縮

**具体例（Qwen2.5-VL, 224x224入力）:**
- 1画像: `grid_t=1, grid_h=16, grid_w=16` → `1 × 16 × 16 / 4 = 64トークン`
- 10フレーム動画: `grid_t=5, grid_h=16, grid_w=16` → `5 × 16 × 16 / 4 = 320トークン`
- 30フレーム動画: `grid_t=15, grid_h=16, grid_w=16` → `15 × 16 × 16 / 4 = 960トークン`
- 100フレーム動画: `grid_t=50, grid_h=16, grid_w=16` → `50 × 16 × 16 / 4 = 3,200トークン`

**全てのフレームのトークンがLLMに渡される。過去フレームのドロップは一切ない。**

### 2.5 Qwen3-VLの追加機能

#### DeepStack
- ViTの中間層（layer 8, 16, 24）の特徴量を、LLMデコーダの対応する層に直接注入
- 低レベル（エッジ、テクスチャ）から高レベル（意味的概念）まで多層的な視覚情報をLLMに伝達
- ただし**時間方向の圧縮には寄与しない** -- 各フレームの全トークンが各層で注入される

#### Text-Timestamp Alignment
- Qwen2.5-VLのT-RoPE（時間的回転位置埋め込み）を廃止
- 代わりにテキストベースのタイムスタンプトークン（例: `<3.0 seconds>`）をフレーム群の前に挿入
- 各動画を `grid_thw[:, 0] = 1` にリシェイプし、1フレームずつタイムスタンプで区切る構造に変更

```python
# Qwen3-VLのビデオ処理
video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
video_grid_thw[:, 0] = 1  # 各エントリを単一フレームに分解
```

これにより、Qwen3-VLでは動画が「タイムスタンプ付きの独立画像の列」として処理される。

---

## 3. MEM（Multi-Scale Embodied Memory）のアーキテクチャ詳細

### 3.1 Space-Time Separable Attention

MEMの核心的な革新は、標準的なViTのattentionパターンを改変し、**空間attentionと時間attentionを分離・交互配置**する点にある。

**構造:**
- **通常層（大部分）:** 各フレーム内の全パッチに対する双方向spatial attention（標準ViTと同一）
- **4層ごと:** 上記に加えて、**同一空間位置のパッチを時間方向にcausal attentionで接続**

**計算量の比較:**
- ナイーブ（全パッチ×全フレームにjoint attention）: O(n²K²)  ←実用不可能
- MEM（分離attention）: O(Kn² + nK²)  ←劇的に効率化

ここで n = 空間パッチ数、K = フレーム数。

### 3.2 過去フレームトークンのドロップ

MEMの最も重要な設計:

> "only passes the representation computed for the current timestep onwards (dropping representations for all patches from past timesteps)"

- ViTの上位層で、**現在フレーム以外の全パッチ表現を破棄**
- LLMバックボーンには**現在フレームのトークンのみ**が渡される
- しかしそのトークンは、時間方向のcausal attentionにより**過去フレームの情報で enriched（強化）されている**
- 結果: **LLMに渡されるトークン数 = 単一フレームVLAと同一**

### 3.3 学習可能パラメータの追加なし

> "does not introduce new learnable parameters compared to standard, single-image ViTs. Video encoding capabilities are added by modifying the attention pattern."

- 標準ViTの重みをそのまま初期化に使用可能
- 時間位置埋め込みには固定正弦波を使用（e(0)=0で単一フレーム互換性を維持）
- 事前学習済みVLMのViT重みから出発可能

### 3.4 MEMの効果

- 事前学習時6フレーム、post-training時18フレームまで拡張
- 推論レイテンシが単一フレームVLAとほぼ同等
- 最大15分間のタスク（キッチン片付け等）を実行可能

---

## 4. 比較分析

### 4.1 アーキテクチャ比較表

| 特徴 | Qwen3-VL (ネイティブ) | MEM |
|------|----------------------|-----|
| **ViT内フレーム間attention** | なし（3D Convのみ） | あり（4層ごとにcausal temporal attention） |
| **LLMへのトークン** | 全フレームの全トークン | 現在フレームのトークンのみ |
| **時間方向の圧縮** | temporal_patch_size=2で2フレーム→1（Conv層のみ） | ViT上位層で過去フレームを全ドロップ |
| **LLMトークン数（10フレーム時）** | 320トークン（224x224） | 64トークン（単一フレーム相当） |
| **LLMトークン数（100フレーム時）** | 3,200トークン | 64トークン（変わらず） |
| **時系列情報のLLMへの伝達** | LLMが全フレームトークンを直接処理 | ViTが時系列情報を現在フレームトークンに圧縮してからLLMへ |
| **追加パラメータ** | -- | なし（attentionパターン変更のみ） |
| **ViT事前学習重み** | 専用学習が必要 | 標準VLMのViT重みから初期化可能 |

### 4.2 VLA応用における致命的差異

#### Qwen3-VLをそのまま使う場合の問題

VLAでは典型的に:
- **制御周波数**: 5-50Hz（1秒に5-50フレーム）
- **必要な時間文脈**: 数秒〜数分（数十〜数千フレーム）
- **推論レイテンシ要件**: 数十ms以下

例えば、5Hzで10秒間の文脈（50フレーム、224x224）をQwen3-VLに渡す場合:
```
grid_t = 50 / 2 = 25
tokens = 25 × 14 × 14 / 4 = 1,225トークン (patch_size=16の場合)
```

これに対してMEMなら:
```
tokens = 1 × 14 × 14 / 4 ≈ 49トークン（現在フレームのみ）
```

**約25倍のトークン数差**が生じる。LLMのself-attentionはトークン数の二乗に比例するため、計算コスト差はさらに大きくなる。

さらに深刻なのは、Qwen3-VLでは**フレーム間の時系列理解をLLMが全て担う**点である。ViTはフレームを独立に処理するため、「物体の動き」「状態変化」「因果関係」といった時間的推論はLLMの長いコンテキスト上のattentionに完全に依存する。MEMではこれをViT内で効率的に処理済みである。

---

## 5. 改造の方向性

Qwen3-VLベースのVLAにMEM的な時系列処理を導入するには、以下の改造が必要:

### 5.1 最小限の改造案

1. **Temporal Attention層の追加**: Qwen3-VLのViT（27層）の4層ごと（layer 4, 8, 12, 16, 20, 24）にcausal temporal attentionを挿入
   - 同一空間位置のパッチを時間軸方向にattend
   - 新規パラメータは不要（既存のattention重みを再利用、パターンのみ変更）

2. **過去フレームトークンのドロップ**: ViT上位層（例: layer 20以降）で現在フレーム以外のトークンを破棄

3. **時間位置埋め込みの追加**: 固定正弦波の時間位置埋め込みを追加（e(0)=0で単一画像互換性維持）

### 5.2 Qwen3-VL固有の考慮事項

- **DeepStackとの整合**: 中間層（layer 8, 16, 24）でLLMに注入される特徴にも時間情報が含まれるよう、temporal attention層の配置を調整
- **Text-Timestamp Alignmentの廃止/変更**: VLAでは動画を「独立画像の列」として扱うQwen3-VL方式よりも、MEMの連続的な時間処理の方が適切
- **Window Attentionとの共存**: spatial attentionはwindow attention方式を維持し、temporal attentionのみ追加するハイブリッド構造

### 5.3 実装上の注意

- `cu_seqlens`による各フレームのattention分離を、temporal attention層では意図的に解除する必要がある
- Qwen3-VLの3D Convパッチ埋め込みはそのまま活用可能（2フレーム局所融合 + ViT内temporal attentionでグローバル融合）
- 事前学習: ロボットデータに加え、通常の動画データでのpre-trainingが有効（MEM論文の知見）

---

## 6. 結論

### Qwen3.5のネイティブ動画処理でMEM的な時系列情報をLLMに渡せるか、それとも改造が必要か

**改造が必要である。** 理由は明確:

1. **Qwen3-VLのViTにはフレーム間attentionが存在しない。** 唯一の時間融合は入力段の3D Conv（2フレーム局所結合）のみであり、これは数秒〜数分の時間文脈を捉えるには全く不十分。

2. **全フレームトークンがLLMに渡される設計である。** VLAの実時間制御で求められるフレーム数（数十〜数百）では、LLMのコンテキスト長とレイテンシの両面で非現実的。

3. **時系列理解を全てLLMに委ねる設計である。** Qwen3-VLは「画像の列+タイムスタンプ」としてLLMに丸投げするため、時系列パターン認識の効率がMEMに比べて著しく劣る。

**推奨アプローチ:** Qwen3-VLのViTにMEMのSpace-Time Separable Attentionを移植し、過去フレームトークンのドロップ機構を追加する。これにより、Qwen3-VLのDeepStack等の利点を活かしつつ、VLAに必要な効率的時系列処理を実現できる。追加の学習可能パラメータは不要であり、attentionパターンの変更のみで実装可能。

---

## Sources

- [Qwen2.5-VL Technical Report (arXiv 2502.13923)](https://arxiv.org/abs/2502.13923)
- [Qwen3-VL Technical Report (arXiv 2511.21631)](https://arxiv.org/abs/2511.21631)
- [MEM: Multi-Scale Embodied Memory (arXiv 2603.03596)](https://arxiv.org/abs/2603.03596)
- [Qwen3-VL GitHub Repository](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3-VL GitHub Issue #753: Vision Attention Masks](https://github.com/QwenLM/Qwen3-VL/issues/753)
- [HuggingFace Transformers: Qwen2.5-VL modeling](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)
- [HuggingFace Transformers: Qwen3-VL modeling](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)
- [Qwen2.5-VL Code Walkthrough (Towards AI)](https://towardsai.net/p/machine-learning/qwen2-5-vl-a-hands-on-code-walkthrough)
- [Qwen2-VL Code Walkthrough (Medium)](https://medium.com/data-science-collective/qwen2-vl-a-hands-on-code-walkthrough-c5a4e073e9b3)
- [Qwen2.5-VL Blog Post](https://qwenlm.github.io/blog/qwen2.5-vl/)
- [Physical Intelligence MEM Project Page](https://www.pi.website/research/memory)
- [DeepWiki: Qwen2.5-VL Architecture](https://deepwiki.com/QwenLM/Qwen2.5-VL/2-model-architecture)
