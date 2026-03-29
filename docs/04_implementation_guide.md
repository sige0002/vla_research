# 具体的な実装手順書

**作成日**: 2026年3月29日

---

## 目次

1. [全体アーキテクチャ詳細設計](#1-全体アーキテクチャ詳細設計)
2. [ベースVLA実装](#2-ベースvla実装)
3. [MEMメモリ統合実装](#3-memメモリ統合実装)
4. [C-RADIO ViT置換実装](#4-c-radio-vit置換実装)
5. [RLT実装](#5-rlt実装)
6. [RECAP実装](#6-recap実装)
7. [学習レシピ](#7-学習レシピ)
8. [推論パイプライン](#8-推論パイプライン)

---

## 1. 全体アーキテクチャ詳細設計

### 1.1 モジュール構成

```
project/
├── models/
│   ├── vision/
│   │   ├── c_radio_encoder.py      # C-RADIOラッパー
│   │   ├── qwen_vit.py             # Qwen ViTラッパー
│   │   └── video_encoder.py        # MEM短期メモリ（時間アテンション）
│   ├── vlm/
│   │   ├── qwen_vlm.py             # Qwen VLM統合
│   │   └── projector.py            # ViT→LLM射影層
│   ├── policy/
│   │   ├── high_policy.py          # 高ポリシー（VLM + メモリ）
│   │   ├── low_policy.py           # 低ポリシー（アクションエキスパート）
│   │   └── action_expert.py        # DiTフローマッチング
│   ├── memory/
│   │   ├── short_term.py           # 短期ビデオメモリ
│   │   └── long_term.py            # 長期言語メモリ
│   └── rl/
│       ├── rlt.py                  # RLトークン抽出 + Actor-Critic
│       ├── recap.py                # RECAP アドバンテージ条件付け
│       └── value_function.py       # 価値関数
├── training/
│   ├── stage1_alignment.py         # Projectorアライメント
│   ├── stage2_sft.py               # ロボットSFT
│   ├── stage3_recap.py             # RECAP RL学習
│   └── stage4_rlt.py               # RLT精密制御学習
├── data/
│   ├── dataset.py                  # データセットローダー
│   ├── memory_label.py             # メモリラベル生成
│   └── subtask_annotation.py       # サブタスクアノテーション
├── eval/
│   ├── libero_eval.py              # LIBERO評価
│   ├── long_horizon_eval.py        # ロングホライズン評価
│   └── precision_eval.py           # 精密タスク評価
└── configs/
    ├── model.yaml
    ├── train.yaml
    └── eval.yaml
```

### 1.2 データフロー（推論時）

```
Time t:
  Camera images (past K frames) ──► Video Encoder (C-RADIO + MEM temporal attention)
                                          │
                                    Visual tokens (current frame only)
                                          │
  Task goal text ─────────────────────────┤
  Previous memory m_t ────────────────────┤
                                          ▼
                                    Qwen VLM (high policy, 5Hz)
                                          │
                          ┌───────────────┼───────────────┐
                          ▼               ▼               ▼
                    subtask l_{t+1}   memory m_{t+1}   RL tokens
                          │                               │
  Current obs ────────────┤                               │
  Proprioception ─────────┤                               │
                          ▼                               ▼
                    Action Expert                   RLT Actor
                    (Flow Matching)                (action edit)
                          │                               │
                          ▼                               ▼
                    base action ──────────────► final action
                                                          │
                                                 action chunk (10-20 steps)
                                                          │
                                                    Robot (30-100Hz)
```

---

## 2. ベースVLA実装

### 2.1 Qwen VLMの読み込みとカスタマイズ

```python
"""models/vlm/qwen_vlm.py"""
import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

class QwenVLABase(nn.Module):
    """
    Qwen3.5-4Bをベースにしたベース VLAモデル。
    Gated DeltaNet（線形アテンション）+ フルアテンションのハイブリッド構成。
    DeepStack廃止によりViT出力がクリーン → C-RADIO置換が容易。
    VLMの言語出力にサブタスク指示を生成させ、
    別途アクションエキスパートでロボット行動を生成する。
    """
    def __init__(self, model_name="Qwen/Qwen3.5-4B-Instruct"):
        super().__init__()
        # VLMロード
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

        # アクションエキスパート（DiTフローマッチング）
        self.action_expert = FlowMatchingActionExpert(
            condition_dim=self.vlm.config.hidden_size,  # 2048
            action_dim=7,          # 6DoF + gripper
            chunk_size=10,         # 10ステップ先まで予測
            dit_layers=16,
            dit_hidden=512,
        )

    def forward(self, images, text, proprioception, noisy_actions, timestep):
        # VLM forward（画像+テキスト → 内部表現）
        vlm_outputs = self.vlm(
            pixel_values=images,
            input_ids=text,
            output_hidden_states=True,
        )
        # 最終隠れ状態をアクションエキスパートの条件として使用
        condition = vlm_outputs.hidden_states[-1]  # (B, seq, 2048)

        # KVキャッシュの最終16層を取得（Xiaomi-R0方式）
        kv_cache = vlm_outputs.past_key_values[-16:]

        # アクションエキスパート forward
        predicted_actions = self.action_expert(
            kv_cache=kv_cache,
            proprioception=proprioception,
            noisy_actions=noisy_actions,
            timestep=timestep,
        )
        return predicted_actions
```

### 2.2 フローマッチング アクションエキスパート

```python
"""models/policy/action_expert.py"""
import torch
import torch.nn as nn
import math

class FlowMatchingActionExpert(nn.Module):
    """
    DiTベースのフローマッチング アクションエキスパート。
    Xiaomi-R0の設計を参考にした実装。
    """
    def __init__(
        self,
        condition_dim=2048,
        action_dim=7,
        chunk_size=10,
        dit_layers=16,
        dit_hidden=512,
        num_heads=8,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        # 固有受容覚エンコーダ
        self.proprio_encoder = nn.Sequential(
            nn.Linear(action_dim, dit_hidden),
            nn.SiLU(),
            nn.Linear(dit_hidden, dit_hidden),
        )

        # ノイズ付き行動の射影
        self.action_proj = nn.Linear(action_dim, dit_hidden)

        # タイムステップ埋め込み（正弦波）
        self.time_embed = SinusoidalTimestepEmbedding(dit_hidden)

        # Attention Sinkトークン（Xiaomi-R0の革新）
        self.sink_tokens = nn.Parameter(torch.randn(4, dit_hidden) * 0.02)

        # DiTブロック
        self.dit_blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=dit_hidden,
                num_heads=num_heads,
                condition_dim=condition_dim,
            )
            for _ in range(dit_layers)
        ])

        # 出力射影（ベロシティ予測）
        self.output_proj = nn.Sequential(
            nn.LayerNorm(dit_hidden),
            nn.Linear(dit_hidden, action_dim),
        )

    def forward(self, kv_cache, proprioception, noisy_actions, timestep):
        """
        Args:
            kv_cache: VLMの最終16層のKVキャッシュ
            proprioception: (B, action_dim) 現在の固有受容覚
            noisy_actions: (B, chunk_size, action_dim) ノイズ付き行動
            timestep: (B,) フロー・タイムステップ τ ∈ [0, 1]
        Returns:
            velocity: (B, chunk_size, action_dim) 予測ベロシティ
        """
        B = noisy_actions.shape[0]

        # トークン準備
        action_tokens = self.action_proj(noisy_actions)  # (B, chunk, hidden)
        proprio_token = self.proprio_encoder(proprioception).unsqueeze(1)  # (B, 1, hidden)
        time_emb = self.time_embed(timestep)  # (B, hidden)
        sink = self.sink_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, 4, hidden)

        # トークン結合
        tokens = torch.cat([sink, proprio_token, action_tokens], dim=1)
        tokens = tokens + time_emb.unsqueeze(1)

        # DiTブロック（VLM KVキャッシュにクロスアテンション）
        for block in self.dit_blocks:
            tokens = block(tokens, kv_cache)

        # 出力（sink+proprioをスキップ、actionトークンのみ）
        action_out = tokens[:, 5:, :]  # skip sink(4) + proprio(1)
        velocity = self.output_proj(action_out)  # (B, chunk, action_dim)
        return velocity

    def sample(self, kv_cache, proprioception, num_steps=10):
        """
        推論時: フロー・マッチングサンプリング（前方オイラー法）。
        """
        B = proprioception.shape[0]
        # 純ノイズから開始
        x = torch.randn(B, self.chunk_size, self.action_dim,
                        device=proprioception.device)

        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=x.device)
            v = self.forward(kv_cache, proprioception, x, t)
            x = x + v * dt

        return x  # クリーンなアクションチャンク


class DiTBlock(nn.Module):
    """DiT Transformer Block with cross-attention to VLM KV cache."""
    def __init__(self, hidden_dim, num_heads, condition_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True,
            kdim=condition_dim, vdim=condition_dim,
        )

        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x, kv_cache):
        # Self-attention (Lambda-Shape mask適用可能)
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h)
        x = x + h

        # Cross-attention to VLM features
        h = self.norm2(x)
        # kv_cacheからkey, valueを取り出して使用
        h, _ = self.cross_attn(h, kv_cache, kv_cache)
        x = x + h

        # FFN
        h = self.norm3(x)
        x = x + self.ffn(h)
        return x


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.proj(emb)
```

### 2.3 フローマッチング損失

```python
"""training/losses.py"""
import torch
import torch.nn.functional as F
from torch.distributions import Beta

def flow_matching_loss(model, kv_cache, proprioception, clean_actions):
    """
    条件付きフローマッチング損失。
    pi0の学習手法に従い、Beta分布で低タイムステップを重視。

    Args:
        clean_actions: (B, chunk_size, action_dim) グラウンドトゥルース
    """
    B = clean_actions.shape[0]
    device = clean_actions.device

    # タイムステップをBeta分布からサンプル（低τを重視）
    # pi0: p(τ) = Beta((s-τ)/s; 1.5, 1) → 低ノイズレベルを重視
    beta_dist = Beta(torch.tensor(1.5), torch.tensor(1.0))
    t = beta_dist.sample((B,)).to(device)

    # ノイズ
    noise = torch.randn_like(clean_actions)

    # 線形補間: x_t = (1-t) * noise + t * clean_actions
    t_expand = t[:, None, None]
    noisy_actions = (1 - t_expand) * noise + t_expand * clean_actions

    # ターゲット ベロシティ: v* = clean_actions - noise
    target_velocity = clean_actions - noise

    # モデル予測
    pred_velocity = model(kv_cache, proprioception, noisy_actions, t)

    # MSE損失
    loss = F.mse_loss(pred_velocity, target_velocity)
    return loss
```

---

## 3. MEMメモリ統合実装

### 3.1 短期ビデオメモリ（空間-時間分離アテンション）

```python
"""models/memory/short_term.py"""
import torch
import torch.nn as nn
import math

class SpatioTemporalAttention(nn.Module):
    """
    MEM論文の空間-時間分離アテンション。
    既存ViTの各レイヤに追加する時間アテンションモジュール。
    """
    def __init__(self, hidden_dim, num_heads, max_frames=18):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        # 正弦波時間位置エンコーディング
        self.register_buffer(
            'temporal_pos_enc',
            self._sinusoidal_encoding(max_frames, hidden_dim)
        )

    def _sinusoidal_encoding(self, max_len, dim):
        """t=0で値0（単一画像と等価）"""
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # t=0で全て0にするためのオフセット（単一画像時に影響なし）
        pe[0, :] = 0.0
        return pe

    def forward(self, x, num_frames):
        """
        Args:
            x: (B, K*N, D) 全フレーム×全パッチの特徴
               K=フレーム数, N=パッチ数/フレーム
            num_frames: int フレーム数K
        Returns:
            x: (B, K*N, D) 時間アテンション適用後
        """
        B, total_patches, D = x.shape
        N = total_patches // num_frames  # パッチ数/フレーム

        # 空間位置ごとに時間アテンション
        # reshape: (B, K, N, D) → (B*N, K, D) で時間軸にアテンション
        x = x.view(B, num_frames, N, D)
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, N, K, D)
        x = x.view(B * N, num_frames, D)

        # 時間位置エンコーディング追加
        pos = self.temporal_pos_enc[:num_frames].unsqueeze(0)
        x_with_pos = x + pos

        # 因果的時間アテンション（過去のフレームのみ参照）
        causal_mask = torch.triu(
            torch.ones(num_frames, num_frames, device=x.device) * float('-inf'),
            diagonal=1
        )
        attn_out, _ = self.temporal_attn(
            x_with_pos, x_with_pos, x_with_pos,
            attn_mask=causal_mask
        )
        x = x + attn_out  # 残差接続

        # reshape戻し: (B*N, K, D) → (B, K*N, D)
        x = x.view(B, N, num_frames, D)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, total_patches, D)

        return x


class MEMVideoEncoder(nn.Module):
    """
    MEMのビデオエンコーダ。
    既存ViTに空間-時間分離アテンションを4層ごとに挿入。
    上層で過去フレームのパッチ表現をドロップ。
    """
    def __init__(self, base_vit, temporal_interval=4, drop_past_layer=20):
        """
        Args:
            base_vit: Qwen ViTまたはC-RADIOのViT
            temporal_interval: 時間アテンション挿入間隔（4層ごと）
            drop_past_layer: このレイヤ以降、過去フレームのパッチをドロップ
        """
        super().__init__()
        self.base_vit = base_vit
        self.temporal_interval = temporal_interval
        self.drop_past_layer = drop_past_layer

        # 既存ViTの各レイヤの後に時間アテンションを挿入
        hidden_dim = base_vit.config.hidden_size
        num_heads = base_vit.config.num_attention_heads
        num_layers = base_vit.config.num_hidden_layers

        self.temporal_attns = nn.ModuleDict()
        for i in range(0, num_layers, temporal_interval):
            self.temporal_attns[str(i)] = SpatioTemporalAttention(
                hidden_dim, num_heads
            )

    def forward(self, pixel_values_sequence):
        """
        Args:
            pixel_values_sequence: (B, K, C, H, W) K枚のフレーム
        Returns:
            features: (B, N, D) 現在フレームの視覚特徴のみ
        """
        B, K, C, H, W = pixel_values_sequence.shape

        # 全フレームをパッチに変換
        # (B, K, C, H, W) → (B*K, C, H, W) → ViT patch embedding
        frames = pixel_values_sequence.view(B * K, C, H, W)
        patch_embeds = self.base_vit.patch_embed(frames)  # (B*K, N, D)
        N, D = patch_embeds.shape[1], patch_embeds.shape[2]
        patch_embeds = patch_embeds.view(B, K * N, D)

        # ViTレイヤを順次処理
        x = patch_embeds
        for i, layer in enumerate(self.base_vit.encoder.layers):
            # 空間アテンション（標準ViTレイヤ）
            x = layer(x)

            # 時間アテンション（4層ごと）
            if str(i) in self.temporal_attns:
                x = self.temporal_attns[str(i)](x, num_frames=K)

            # 上層で過去フレームをドロップ
            if i == self.drop_past_layer:
                x = x.view(B, K, N, D)
                x = x[:, -1, :, :]  # 現在フレームのみ (B, N, D)
                K = 1  # 以降は1フレーム

        return x  # (B, N, D) 現在フレームの特徴のみ
```

### 3.2 長期言語メモリ

```python
"""models/memory/long_term.py"""
import torch
import torch.nn as nn
from typing import Optional

class LongTermLanguageMemory:
    """
    MEMの長期言語メモリ管理。
    VLMが生成する自然言語テキストとしてメモリを管理。
    """
    def __init__(self, max_memory_tokens=256):
        self.max_memory_tokens = max_memory_tokens
        self.current_memory = ""  # 初期状態：空文字列

    def format_high_policy_input(
        self,
        task_goal: str,
        memory: str,
    ) -> str:
        """
        高ポリシーVLMへの入力プロンプトを構成する。

        Returns:
            formatted_prompt: VLMへの入力テキスト
        """
        prompt = (
            f"Task: {task_goal}\n"
            f"Memory: {memory if memory else 'None'}\n"
            f"Based on the current observation and memory, "
            f"output the next subtask and updated memory.\n"
            f"Format:\n"
            f"Subtask: <next subtask instruction>\n"
            f"Memory: <updated compressed memory>"
        )
        return prompt

    def parse_high_policy_output(self, vlm_output: str):
        """
        高ポリシーVLMの出力からサブタスクとメモリを抽出。
        """
        subtask = ""
        memory = ""
        for line in vlm_output.split("\n"):
            if line.startswith("Subtask:"):
                subtask = line[len("Subtask:"):].strip()
            elif line.startswith("Memory:"):
                memory = line[len("Memory:"):].strip()
        return subtask, memory

    def update(self, new_memory: str, success: bool):
        """
        メモリ更新。成功時のみ更新する（MEM論文の重要知見）。
        ナイーブ連結は20-25%の性能低下を引き起こす。
        """
        if success:
            self.current_memory = new_memory


class HighLevelPolicy(nn.Module):
    """
    高レベルポリシー：VLM + 長期メモリ + 短期ビデオメモリ。
    5Hzで実行し、サブタスク指示とメモリ更新を生成する。
    """
    def __init__(self, vlm, video_encoder, tokenizer):
        super().__init__()
        self.vlm = vlm
        self.video_encoder = video_encoder
        self.tokenizer = tokenizer
        self.memory_manager = LongTermLanguageMemory()

    def forward(self, video_frames, task_goal, memory_text):
        """
        Args:
            video_frames: (B, K, C, H, W) 直近Kフレーム
            task_goal: str タスク目標
            memory_text: str 前回のメモリ
        Returns:
            subtask: str 次のサブタスク指示
            updated_memory: str 更新されたメモリ
            visual_features: tensor 低ポリシーへの条件
        """
        # 短期ビデオメモリ（空間-時間分離アテンション）
        visual_features = self.video_encoder(video_frames)

        # プロンプト構成
        prompt = self.memory_manager.format_high_policy_input(
            task_goal, memory_text
        )

        # VLM推論（サブタスク + メモリ生成）
        inputs = self.tokenizer(prompt, return_tensors="pt")
        vlm_output = self.vlm.generate(
            pixel_values=visual_features,
            **inputs,
            max_new_tokens=128,
        )
        output_text = self.tokenizer.decode(vlm_output[0])

        # パース
        subtask, updated_memory = self.memory_manager.parse_high_policy_output(
            output_text
        )

        return subtask, updated_memory, visual_features
```

### 3.3 メモリラベル生成パイプライン

```python
"""data/memory_label.py"""

def generate_memory_labels(episode, subtask_annotations, llm):
    """
    学習用のメモリラベルを生成するパイプライン。
    MEM論文のアプローチに従う。

    Args:
        episode: エピソードデータ（画像系列、アクション系列）
        subtask_annotations: サブタスク区切りのアノテーション
        llm: メモリ圧縮用のLLM（Qwen等）
    Returns:
        memory_labels: 各タイムステップでのメモリラベル
    """
    memory_labels = []
    accumulated_events = []

    for i, subtask in enumerate(subtask_annotations):
        # サブタスク完了時にイベントを追加
        accumulated_events.append(subtask['description'])

        # LLMでメモリを圧縮
        compression_prompt = (
            f"以下の完了したサブタスクを、将来のタスク実行に関連する"
            f"情報だけを残して圧縮してください。\n"
            f"冗長な詳細は削除してください（例：色の列挙→個数のみ）。\n\n"
            f"完了タスク:\n"
            + "\n".join(f"- {e}" for e in accumulated_events)
            + f"\n\n全体タスク目標: {episode['task_goal']}\n"
            f"圧縮メモリ:"
        )

        compressed_memory = llm.generate(compression_prompt, max_tokens=64)
        memory_labels.append({
            'timestep': subtask['end_timestep'],
            'subtask': subtask['description'],
            'memory': compressed_memory,
        })

    return memory_labels
```

---

## 4. C-RADIO ViT置換実装

### 4.1 C-RADIOエンコーダラッパー

```python
"""models/vision/c_radio_encoder.py"""
import torch
import torch.nn as nn

class CRadioVisionEncoder(nn.Module):
    """
    C-RADIOv4-SO400Mを VLA のビジョンエンコーダとして使用するラッパー。
    """
    def __init__(self, model_name="nvidia/C-RADIOv4-SO400M"):
        super().__init__()
        # C-RADIOモデルのロード
        self.radio = torch.hub.load(
            'NVlabs/RADIO', 'radio_model',
            version='radio_v2.5-l',  # or C-RADIOv4
            adaptor_names=['siglip2-g'],  # アダプタ指定
        )
        self.radio.eval()  # 初期は凍結

        # C-RADIO config
        self.hidden_size = 1152  # SO400M embed_dim
        self.patch_size = 16

    def forward(self, pixel_values, return_adaptor=False):
        """
        Args:
            pixel_values: (B, C, H, W) 画像入力 [0, 1]
        Returns:
            summary: (B, D) 画像全体の要約特徴
            spatial: (B, N, D) 空間特徴マップ
        """
        # C-RADIO forward
        summary, spatial = self.radio(pixel_values)
        # summary: (B, 1152) CLSトークン相当
        # spatial: (B, N, 1152) パッチごとの空間特徴

        if return_adaptor:
            # SigLIP2アダプタを通した出力も取得可能
            adaptor_output = self.radio.forward_adaptors(summary, spatial)
            return summary, spatial, adaptor_output

        return summary, spatial


class CRadioProjector(nn.Module):
    """
    C-RADIO出力をQwen LLMの入力空間にマッピングする射影層。
    Spatial Merge (2x2) を含む。
    """
    def __init__(
        self,
        radio_dim=1152,
        llm_dim=2560,      # Qwen3-VL-4B
        spatial_merge=2,    # 2x2パッチ統合
    ):
        super().__init__()
        self.spatial_merge = spatial_merge
        merged_dim = radio_dim * spatial_merge * spatial_merge

        self.proj = nn.Sequential(
            nn.Linear(merged_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
        self.norm = nn.LayerNorm(llm_dim)

    def forward(self, spatial_features, h_patches, w_patches):
        """
        Args:
            spatial_features: (B, H*W, D) C-RADIOの空間特徴
            h_patches, w_patches: パッチグリッドのサイズ
        Returns:
            projected: (B, H'*W', llm_dim) マージ+射影後のトークン
        """
        B, N, D = spatial_features.shape
        # reshape to spatial grid
        x = spatial_features.view(B, h_patches, w_patches, D)

        # Spatial Merge: 2x2パッチを結合
        s = self.spatial_merge
        h_new = h_patches // s
        w_new = w_patches // s
        x = x.view(B, h_new, s, w_new, s, D)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, h_new * w_new, s * s * D)

        # 射影
        x = self.proj(x)
        x = self.norm(x)
        return x
```

### 4.2 統合VLAモデル（C-RADIO版）

```python
"""models/vlm/cradio_qwen_vla.py"""
import torch
import torch.nn as nn

class CRadioQwenVLA(nn.Module):
    """
    C-RADIO ViT + Qwen LLM + MEM + Flow Matching Action Expert
    の統合VLAモデル。
    """
    def __init__(self, config):
        super().__init__()
        # Vision Encoder
        self.vision_encoder = CRadioVisionEncoder()
        self.projector = CRadioProjector(
            radio_dim=1152,
            llm_dim=config.llm_dim,
        )

        # MEM Video Encoder（C-RADIO上に構築）
        self.video_encoder = MEMVideoEncoder(
            base_vit=self.vision_encoder,
            temporal_interval=4,
        )

        # LLM (Qwen, ViT部分は除外して読み込み)
        self.llm = load_qwen_llm_only(config.qwen_model_name)

        # Action Expert
        self.action_expert = FlowMatchingActionExpert(
            condition_dim=config.llm_dim,
            action_dim=config.action_dim,
            chunk_size=config.chunk_size,
        )

        # Memory
        self.memory_manager = LongTermLanguageMemory()

    def forward_high_policy(self, video_frames, task_goal, memory):
        """高ポリシー: 5Hzで実行"""
        # C-RADIO + MEMビデオエンコーダ
        visual_tokens = self.video_encoder(video_frames)
        # Projector
        projected = self.projector(visual_tokens, h, w)
        # LLM（サブタスク+メモリ生成）
        subtask, new_memory, hidden_states = self.llm_forward(
            projected, task_goal, memory
        )
        return subtask, new_memory, hidden_states

    def forward_low_policy(self, hidden_states, proprioception,
                           noisy_actions, timestep):
        """低ポリシー: 30-100Hzで実行（Action Chunkingで非同期）"""
        actions = self.action_expert(
            hidden_states, proprioception, noisy_actions, timestep
        )
        return actions
```

---

## 5. RLT実装

### 5.1 RLトークン抽出器

```python
"""models/rl/rlt.py"""
import torch
import torch.nn as nn

class RLTokenExtractor(nn.Module):
    """
    VLAの内部表現からRLトークンを抽出する情報ボトルネック。
    エンコーダで圧縮し、デコーダで再構成する。
    """
    def __init__(
        self,
        vla_dim=2560,       # VLA内部表現の次元
        rl_token_dim=256,   # RLトークンの次元
        num_rl_tokens=8,    # RLトークンの数
        num_layers=4,
    ):
        super().__init__()
        self.num_rl_tokens = num_rl_tokens

        # 学習可能なクエリトークン
        self.query_tokens = nn.Parameter(
            torch.randn(num_rl_tokens, rl_token_dim) * 0.02
        )

        # エンコーダ: VLA埋め込み → RLトークン
        self.encoder_proj = nn.Linear(vla_dim, rl_token_dim)
        self.encoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=rl_token_dim, nhead=8, batch_first=True
            ),
            num_layers=num_layers,
        )

        # デコーダ: RLトークン → VLA埋め込みの再構成
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=rl_token_dim, nhead=8, batch_first=True
            ),
            num_layers=num_layers,
        )
        self.decoder_proj = nn.Linear(rl_token_dim, vla_dim)

    def encode(self, vla_embeddings):
        """
        Args:
            vla_embeddings: (B, seq, vla_dim) VLAの内部埋め込み
        Returns:
            rl_tokens: (B, num_rl_tokens, rl_token_dim)
        """
        B = vla_embeddings.shape[0]
        memory = self.encoder_proj(vla_embeddings)
        queries = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        rl_tokens = self.encoder(queries, memory)
        return rl_tokens

    def decode(self, rl_tokens, target_seq_len):
        """再構成（事前学習用）"""
        B = rl_tokens.shape[0]
        target_queries = torch.zeros(B, target_seq_len, rl_tokens.shape[-1],
                                     device=rl_tokens.device)
        reconstructed = self.decoder(target_queries, rl_tokens)
        return self.decoder_proj(reconstructed)

    def pretrain_loss(self, vla_embeddings):
        """再構成損失で事前学習"""
        rl_tokens = self.encode(vla_embeddings)
        reconstructed = self.decode(rl_tokens, vla_embeddings.shape[1])
        return nn.functional.mse_loss(reconstructed, vla_embeddings.detach())


class RLTActorCritic(nn.Module):
    """
    RLトークン上で動作する小規模Actor-Critic。
    VLA予測アクションを「編集」する設計。
    """
    def __init__(
        self,
        rl_token_dim=256,
        num_rl_tokens=8,
        action_dim=7,
        chunk_size=10,
        ref_dropout=0.1,    # Reference-action dropout
    ):
        super().__init__()
        self.ref_dropout = ref_dropout
        input_dim = rl_token_dim * num_rl_tokens + action_dim * chunk_size

        # Actor: RLトークン + VLA予測アクション → アクション編集量
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, action_dim * chunk_size),
            nn.Tanh(),  # 編集量を制限
        )
        self.edit_scale = nn.Parameter(torch.tensor(0.1))  # 初期は小さい編集

        # Critic: RLトークン → 状態価値
        self.critic = nn.Sequential(
            nn.Linear(rl_token_dim * num_rl_tokens, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(self, rl_tokens, vla_actions, deterministic=False):
        """
        Args:
            rl_tokens: (B, num_tokens, rl_token_dim) RLトークン
            vla_actions: (B, chunk_size, action_dim) VLA予測アクション
        Returns:
            edited_actions: (B, chunk_size, action_dim) 編集後のアクション
            value: (B, 1) 状態価値
        """
        B = rl_tokens.shape[0]
        rl_flat = rl_tokens.view(B, -1)
        vla_flat = vla_actions.view(B, -1)

        # Reference-action dropout（学習時のみ）
        if self.training and not deterministic:
            mask = torch.rand(B, 1, device=vla_flat.device) > self.ref_dropout
            vla_flat = vla_flat * mask

        # Actor: アクション編集量を予測
        actor_input = torch.cat([rl_flat, vla_flat], dim=-1)
        edit = self.actor(actor_input).view(B, -1, vla_actions.shape[-1])
        edited_actions = vla_actions + edit * self.edit_scale

        # Critic
        value = self.critic(rl_flat)

        return edited_actions, value
```

---

## 6. RECAP実装

### 6.1 アドバンテージ条件付け

```python
"""models/rl/recap.py"""
import torch
import torch.nn as nn

class RECAPAdvantageConditioning:
    """
    RECAP法のアドバンテージ条件付けモジュール。
    VLMのテキスト入力にアドバンテージ情報を追加する。
    """
    def format_prompt_with_advantage(
        self,
        task_goal: str,
        memory: str,
        advantage: str = None,  # "positive", "negative", or None
    ) -> str:
        """推論時/学習時のプロンプト構成"""
        prompt = f"Task: {task_goal}\n"
        if memory:
            prompt += f"Memory: {memory}\n"
        if advantage:
            prompt += f"Advantage: {advantage}\n"
        prompt += "Subtask:"
        return prompt

    def classifier_free_guidance(
        self,
        model,
        inputs,
        beta=2.0,    # ガイダンス強度
    ):
        """
        推論時の分類器フリーガイダンス。
        π*(a|o,l) ∝ π_ref(a|o,l) × (π_ref(a|I,o,l) / π_ref(a|o,l))^β

        conditional = advantage付き、unconditional = advantageなし
        """
        # Conditional forward（Advantage: positive付き）
        cond_inputs = self.format_prompt_with_advantage(
            inputs['task_goal'], inputs['memory'], advantage="positive"
        )
        log_prob_cond = model.log_prob(cond_inputs, inputs['images'])

        # Unconditional forward（Advantageなし）
        uncond_inputs = self.format_prompt_with_advantage(
            inputs['task_goal'], inputs['memory'], advantage=None
        )
        log_prob_uncond = model.log_prob(uncond_inputs, inputs['images'])

        # ガイダンス適用
        guided_log_prob = log_prob_uncond + beta * (log_prob_cond - log_prob_uncond)
        return guided_log_prob


class DistributionalValueFunction(nn.Module):
    """
    RECAP用の分布的価値関数。
    201ビンでタスク完了までのステップ数の分布を予測。
    """
    def __init__(self, input_dim=2560, num_bins=201, max_steps=1000):
        super().__init__()
        self.num_bins = num_bins
        self.max_steps = max_steps

        # 小規模VLMまたはMLP
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, num_bins),
        )

        # ビンの中心値
        self.register_buffer(
            'bin_centers',
            torch.linspace(0, max_steps, num_bins)
        )

    def forward(self, features):
        """
        Args:
            features: (B, input_dim) VLMの隠れ状態
        Returns:
            value_dist: (B, num_bins) 価値の分布
            expected_value: (B,) 期待値
        """
        logits = self.backbone(features)
        value_dist = torch.softmax(logits, dim=-1)
        expected_value = (value_dist * self.bin_centers.unsqueeze(0)).sum(dim=-1)
        return value_dist, expected_value

    def compute_advantage(self, features_t, features_t1):
        """
        アドバンテージ計算:
        A(s,a) = V(s') - V(s) + 1  (1ステップ進んだので+1)
        """
        _, v_t = self.forward(features_t)
        _, v_t1 = self.forward(features_t1)
        advantage = v_t1 - v_t + 1
        return advantage
```

---

## 7. 学習レシピ

### 7.1 Stage 1: Alignment Pre-training

```yaml
# configs/stage1_alignment.yaml
stage: alignment
model:
  vision_encoder: c_radio  # or qwen_vit
  vision_frozen: true
  llm_frozen: true
  projector_trainable: true

data:
  datasets:
    - name: llava_pretrain     # 画像-テキストペア
      weight: 0.5
    - name: video_caption       # ビデオ-言語タスク（MEM用）
      weight: 0.3
    - name: robot_caption       # ロボット画像キャプション
      weight: 0.2

training:
  learning_rate: 1e-3
  batch_size: 256
  num_steps: 50000
  optimizer: adamw
  warmup_steps: 1000
  weight_decay: 0.0

hardware:
  gpus: 4  # A100 80GB
  precision: bf16
```

### 7.2 Stage 2: Robot SFT

```yaml
# configs/stage2_sft.yaml
stage: robot_sft
model:
  vision_encoder: c_radio
  vision_frozen: false
  vision_lr: 1e-6           # very low lr for ViT
  llm_lora: true
  lora_rank: 64
  projector_trainable: true
  action_expert_trainable: true

data:
  datasets:
    - name: droid
      weight: 0.3
    - name: bridge_v2
      weight: 0.2
    - name: oxe_subset
      weight: 0.2
    - name: memory_tasks      # メモリラベル付きタスク
      weight: 0.2
    - name: vl_data           # 視覚言語データ（知識保持）
      weight: 0.1
  vl_robot_ratio: "1:6"       # Xiaomi-R0方式

training:
  learning_rate: 2e-5
  batch_size: 64
  num_steps: 150000
  optimizer: adamw
  warmup_steps: 5000

  # フローマッチング損失
  flow_matching:
    beta_dist_alpha: 1.5
    beta_dist_beta: 1.0
    denoising_steps: 10

  # メモリ学習
  memory:
    subtask_loss_weight: 1.0
    memory_update_loss_weight: 0.5

hardware:
  gpus: 8  # A100 80GB
  precision: bf16
```

### 7.3 Stage 3: RECAP RL

```yaml
# configs/stage3_recap.yaml
stage: recap_rl

# RECAP specific
recap:
  advantage_conditioning: true
  positive_label: "Advantage: positive"
  negative_label: "Advantage: negative"
  guidance_beta: 2.0

  value_function:
    num_bins: 201
    max_steps: 1000
    lr: 1e-4

  data_collection:
    episodes_per_round: 100
    expert_intervention_rate: 0.1
    autonomous_hours: 6         # 5:30AM - 11:30PM（pi0.6方式）

training:
  learning_rate: 5e-6
  batch_size: 32
  num_rounds: 10
  steps_per_round: 5000

hardware:
  gpus: 4
  robots: 1
```

### 7.4 Stage 4: RLT Fine-tuning

```yaml
# configs/stage4_rlt.yaml
stage: rlt_finetuning

rlt:
  rl_token_dim: 256
  num_rl_tokens: 8
  ref_dropout: 0.1

  # RLトークン抽出器の事前学習
  extractor_pretrain:
    steps: 10000
    lr: 1e-3

  # Actor-Critic学習（実機上）
  actor_critic:
    lr_actor: 3e-4
    lr_critic: 1e-3
    gamma: 0.99
    update_freq: 1             # 毎ステップ更新
    updates_per_step: 100      # オフポリシーで多回更新

  target_tasks:
    - screw_tightening
    - connector_insertion
    - precise_placement

training:
  max_real_time_minutes: 120   # 最大2時間
  min_real_time_minutes: 15    # 最短15分

hardware:
  gpus: 1  # オンボードGPU
  robot: 1
```

---

## 8. 推論パイプライン

### 8.1 非同期階層推論

```python
"""inference/hierarchical_inference.py"""
import torch
import threading
import time
from collections import deque

class HierarchicalVLAInference:
    """
    高ポリシー（5Hz）と低ポリシー（30-100Hz）の非同期推論。
    """
    def __init__(self, model, device='cuda'):
        self.model = model.to(device).eval()
        self.device = device

        # 状態管理
        self.current_subtask = ""
        self.current_memory = ""
        self.action_buffer = deque(maxlen=20)  # アクションチャンクバッファ
        self.frame_buffer = deque(maxlen=18)   # ビデオフレームバッファ

        # RLT Actor-Critic
        self.rlt_extractor = model.rlt_extractor
        self.rlt_actor_critic = model.rlt_actor_critic

        # 非同期制御
        self.high_policy_lock = threading.Lock()
        self.running = False

    def start(self, task_goal):
        """推論ループを開始"""
        self.task_goal = task_goal
        self.running = True
        # 高ポリシーを別スレッドで実行
        self.high_policy_thread = threading.Thread(
            target=self._high_policy_loop
        )
        self.high_policy_thread.start()

    def _high_policy_loop(self):
        """高ポリシーループ（5Hz）"""
        while self.running:
            start_time = time.time()

            if len(self.frame_buffer) >= 6:
                frames = torch.stack(list(self.frame_buffer)).unsqueeze(0)
                with torch.no_grad():
                    subtask, memory, hidden = self.model.forward_high_policy(
                        frames.to(self.device),
                        self.task_goal,
                        self.current_memory,
                    )
                with self.high_policy_lock:
                    self.current_subtask = subtask
                    self.current_memory = memory
                    self._cached_hidden = hidden

            # 5Hzを維持
            elapsed = time.time() - start_time
            sleep_time = max(0, 0.2 - elapsed)  # 200ms間隔
            time.sleep(sleep_time)

    def get_action(self, current_obs, proprioception):
        """
        低ポリシー: 30-100Hzで呼び出される。
        アクションバッファが空の場合、新しいチャンクを生成。
        """
        # フレームバッファに追加
        self.frame_buffer.append(current_obs)

        if len(self.action_buffer) == 0:
            # 新しいアクションチャンクを生成
            with torch.no_grad():
                with self.high_policy_lock:
                    hidden = self._cached_hidden

                # アクションエキスパートでサンプリング
                base_actions = self.model.action_expert.sample(
                    hidden,
                    proprioception.to(self.device),
                    num_steps=10,
                )

                # RLTで精密編集
                rl_tokens = self.rlt_extractor.encode(hidden)
                edited_actions, _ = self.rlt_actor_critic(
                    rl_tokens, base_actions, deterministic=True
                )

                # バッファに追加
                for i in range(edited_actions.shape[1]):
                    self.action_buffer.append(
                        edited_actions[0, i].cpu().numpy()
                    )

        return self.action_buffer.popleft()

    def stop(self):
        self.running = False
        self.high_policy_thread.join()
```

### 8.2 量子化・最適化

```python
"""inference/optimization.py"""

def optimize_for_deployment(model, target_device='rtx4090'):
    """デプロイ向け最適化"""

    # INT4量子化（bfloat16と同等性能、メモリ半減）
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    # C-RADIOはbfloat16のまま（412Mなので量子化不要）
    # Qwen LLMのみINT4量子化
    model.llm = model.llm.quantize(quantization_config)

    # アクションエキスパートはbfloat16
    model.action_expert = model.action_expert.to(torch.bfloat16)

    # RLT Actor-Criticはfloat32（小規模なので）
    # → そのまま

    # torch.compile（PyTorch 2.x最適化）
    model.action_expert = torch.compile(model.action_expert, mode='reduce-overhead')

    return model
```

---

## 補足: 主要パラメータまとめ

| コンポーネント | パラメータ数 | 推論時間（目安） |
|---------------|-------------|-----------------|
| C-RADIO ViT | 412M | ~15ms |
| MEM時間アテンション | ~0 (追加なし) | ~5ms |
| Projector | ~10M | ~1ms |
| Qwen LLM (INT4) | 3-4B | ~100ms |
| Action Expert (DiT) | ~300M | ~20ms |
| RLT Extractor | ~20M | ~2ms |
| RLT Actor-Critic | ~5M | ~1ms |
| **合計（高ポリシー）** | **~4.7B** | **~145ms (≈7Hz)** |
| **合計（低ポリシー、チャンク再利用時）** | - | **~3ms (≈300Hz)** |
