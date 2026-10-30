# PARSeq(Permuted Autoregressive Sequence Models)の推論専用の取り込み。
#
# 出典:
#   - baudm/parseq(Scene Text Recognition Model Hub)
#     Copyright 2022 Darwin Bautista, Licensed under the Apache License, Version 2.0
#     https://github.com/baudm/parseq  (取り込み元: commit 1902db043c029a7e03a3818c616c06600af574be)
#     ライセンス全文: docs/LICENSE-parseq-Apache-2.0.txt
#   - ndl-lab/ndlocr-lite(国立国会図書館, CC BY 4.0)の学習済みモデル ver1.2(parseq-tiny, 高さ 24)の
#     構成(img_size [24, W], tiny: embed_dim 192 / enc 12 層 3 ヘッド / dec 1 層 6 ヘッド)に合わせた。
#     https://github.com/ndl-lab/ndlocr-lite
#
# 変更点(Yayoi OCR):
#   - 学習関連(置換学習・Lightning のシステム・損失)を削除し、推論に必要な部分だけを残した。
#   - timm への依存をなくし、ViT エンコーダ(PatchEmbed + Transformer ブロック)を同じパラメータ名で
#     自前実装した(timm の state_dict と同じキーで読み込める)。
#   - 教師強制による系列の対数尤度 `score_logprob`、列ごとの文字集合マスク、ビームサーチ用の
#     1 ステップ予測 `next_logits`、エンコーダ出力 `encode` を追加した。
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans: int, embed_dim: int) -> None:
        super().__init__()
        self.img_size = tuple(img_size)
        self.patch_size = tuple(patch_size)
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)  # B, N, C


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Encoder(nn.Module):
    """クラストークンなし・分類ヘッドなしの ViT。全パッチのトークンを返す。"""

    def __init__(
        self, img_size, patch_size, embed_dim: int, depth: int, num_heads: int, mlp_ratio: float
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x) + self.pos_embed
        return self.norm(self.blocks(x))


class DecoderLayer(nn.Module):
    """2 ストリーム注意(XLNet)に対応した pre-LN デコーダ層。"""

    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int, layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_q = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_c = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def project_memory(self, memory: Tensor) -> tuple[Tensor, Tensor]:
        """エンコーダ出力 (B, S, C) → cross-attention の K・V(各 (B, H, S, C/H))。デコードの全ステップ・全候補で共有する。"""
        mha = self.cross_attn
        c, h = mha.embed_dim, mha.num_heads
        w, b = mha.in_proj_weight, mha.in_proj_bias
        k = F.linear(memory, w[c : 2 * c], b[c : 2 * c])
        v = F.linear(memory, w[2 * c :], b[2 * c :])
        bsz, s = k.shape[:2]
        return k.view(bsz, s, h, c // h).transpose(1, 2), v.view(bsz, s, h, c // h).transpose(1, 2)

    def cross(self, x: Tensor, kv: tuple[Tensor, Tensor]) -> Tensor:
        """nn.MultiheadAttention と同じ計算(マスクなし)。K・V のバッチは 1 でも x と同じでもよい。"""
        mha = self.cross_attn
        c, h = mha.embed_dim, mha.num_heads
        n, t, _ = x.shape
        q = F.linear(x, mha.in_proj_weight[:c], mha.in_proj_bias[:c])
        q = q.view(n, t, h, c // h).transpose(1, 2)
        k, v = kv
        if k.shape[0] != n:
            k, v = k.expand(n, -1, -1, -1), v.expand(n, -1, -1, -1)
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(n, t, c)
        return F.linear(o, mha.out_proj.weight, mha.out_proj.bias)

    def forward_stream(self, tgt, tgt_norm, tgt_kv, memory_kv, tgt_mask, tgt_key_padding_mask):
        tgt2, _ = self.self_attn(
            tgt_norm, tgt_kv, tgt_kv, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + tgt2
        tgt = tgt + self.cross(self.norm1(tgt), memory_kv)
        return tgt + self.linear2(F.gelu(self.linear1(self.norm2(tgt))))

    def forward(
        self,
        query,
        content,
        memory,
        query_mask,
        content_mask,
        content_key_padding_mask,
        update_content,
    ):
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)
        query = self.forward_stream(
            query, query_norm, content_norm, memory, query_mask, content_key_padding_mask
        )
        if update_content:
            content = self.forward_stream(
                content, content_norm, content_norm, memory, content_mask, content_key_padding_mask
            )
        return query, content


class Decoder(nn.Module):
    def __init__(self, decoder_layer_args: tuple, num_layers: int, embed_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(*decoder_layer_args) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, query, content, memory_kv, query_mask=None, content_mask=None, content_kpm=None
    ):
        """memory_kv: レイヤごとの (K, V) のリスト(`PARSeq.project_memory`)。"""
        for i, mod in enumerate(self.layers):
            last = i == len(self.layers) - 1
            query, content = mod(
                query, content, memory_kv[i], query_mask, content_mask, content_kpm, not last
            )
        return self.norm(query)


class TokenEmbedding(nn.Module):
    def __init__(self, charset_size: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens: Tensor) -> Tensor:
        return math.sqrt(self.embed_dim) * self.embedding(tokens)


class PARSeq(nn.Module):
    """トークン ID: 0 = EOS、1..N = 文字、N+1 = BOS、N+2 = PAD。head は EOS と文字だけを予測する。"""

    def __init__(
        self,
        num_tokens: int,
        max_label_length: int,
        img_size: Sequence[int],
        patch_size: Sequence[int],
        embed_dim: int,
        enc_num_heads: int,
        enc_mlp_ratio: int,
        enc_depth: int,
        dec_num_heads: int,
        dec_mlp_ratio: int,
        dec_depth: int,
        refine_iters: int = 1,
    ) -> None:
        super().__init__()
        self.max_label_length = max_label_length
        self.refine_iters = refine_iters
        # 反復修正のマスク: "onnx" = NDLOCR-Lite 同梱 ONNX の挙動(既定)、"cloze" = baudm/parseq の PyTorch 実装
        self.refine_mask = "onnx"
        self.eos_id = 0
        self.bos_id = num_tokens - 2
        self.pad_id = num_tokens - 1
        self.encoder = Encoder(
            img_size, patch_size, embed_dim, enc_depth, enc_num_heads, enc_mlp_ratio
        )
        self.decoder = Decoder(
            (embed_dim, dec_num_heads, embed_dim * dec_mlp_ratio), dec_depth, embed_dim
        )
        self.head = nn.Linear(embed_dim, num_tokens - 2)  # BOS・PAD は予測しない
        self.text_embed = TokenEmbedding(num_tokens, embed_dim)
        self.pos_queries = nn.Parameter(torch.zeros(1, max_label_length + 1, embed_dim))

    @property
    def device(self) -> torch.device:
        return self.head.weight.device

    def encode(self, img: Tensor) -> Tensor:
        return self.encoder(img)

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor | list,
        tgt_mask: Tensor | None = None,
        tgt_padding_mask: Tensor | None = None,
        tgt_query: Tensor | None = None,
        tgt_query_mask: Tensor | None = None,
    ) -> Tensor:
        """memory: エンコーダ出力(Tensor)か、`project_memory` の結果(K・V のキャッシュ)。"""
        n, length = tgt.shape
        # <bos> は null コンテキスト。BOS より後の文字にだけ位置情報を与える。
        null_ctx = self.text_embed(tgt[:, :1])
        tgt_emb = self.pos_queries[:, : length - 1] + self.text_embed(tgt[:, 1:])
        tgt_emb = torch.cat([null_ctx, tgt_emb], dim=1)
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :length].expand(n, -1, -1)
        kv = memory if isinstance(memory, list) else self.project_memory(memory)
        return self.decoder(tgt_query, tgt_emb, kv, tgt_query_mask, tgt_mask, tgt_padding_mask)

    def project_memory(self, memory: Tensor) -> list[tuple[Tensor, Tensor]]:
        """cross-attention の K・V をレイヤごとに 1 回だけ射影する(全ステップ・全候補で共有)。"""
        return [layer.project_memory(memory) for layer in self.decoder.layers]

    # ------------------------------------------------------------------ 推論

    def forward(
        self, images: Tensor, vocab_mask: Tensor | None = None, refine: bool | None = None
    ) -> Tensor:
        """AR 貪欲デコード + 反復修正(既定 1 回)の logits(N, max_label_length + 1, V)を返す。

        `vocab_mask`(V の bool。True = 許可)があれば、貪欲に選ぶ各ステップで許可外の logits を -inf にする。
        NDLOCR-Lite の ONNX(export 時 decode_ar=True, refine_iters=1)と同じ計算(マスクなしのとき)。
        """
        refine_iters = self.refine_iters if refine is None else int(bool(refine))
        bs = images.shape[0]
        num_steps = self.max_label_length + 1
        memory = self.project_memory(self.encode(images))
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)
        tgt_mask = query_mask = torch.triu(
            torch.ones((num_steps, num_steps), dtype=torch.bool, device=self.device), 1
        )
        tgt_in = torch.full((bs, num_steps), self.pad_id, dtype=torch.long, device=self.device)
        tgt_in[:, 0] = self.bos_id
        logits = []
        for i in range(num_steps):
            j = i + 1
            tgt_out = self.decode(
                tgt_in[:, :j], memory, tgt_mask[:j, :j], tgt_query=pos_queries[:, i:j],
                tgt_query_mask=query_mask[i:j, :j],
            )  # fmt: skip
            p_i = self._head(tgt_out, vocab_mask)
            logits.append(p_i)
            if j < num_steps:
                tgt_in[:, j] = p_i.squeeze(1).argmax(-1)
                if (tgt_in == self.eos_id).any(dim=-1).all():
                    break
        logits = torch.cat(logits, dim=1)
        if refine_iters:
            logits = self._refine(
                logits, memory, pos_queries, query_mask, num_steps, refine_iters, vocab_mask
            )
        return logits

    def _head(self, tgt_out: Tensor, vocab_mask: Tensor | None) -> Tensor:
        """logits。`vocab_mask` は (V) か 行ごとの (N, V)(True = 許可)。EOS(ID 0)は呼び出し側で許可しておく。"""
        out = self.head(tgt_out)
        if vocab_mask is not None:
            m = vocab_mask if vocab_mask.dim() == 1 else vocab_mask[:, None, :]
            out = out.masked_fill(~m, float("-inf"))
        return out

    def _refine(self, logits, memory, pos_queries, query_mask, num_steps, iters, vocab_mask):
        bs = logits.shape[0]
        # 反復修正では「穴埋め(cloze)」マスクを使う。AR の前向きマスクの右側の文脈を開放して得る。
        # 元の実装(baudm/parseq)では tgt_mask と query_mask が同一のテンソルで、ここで両方が変更される。
        tgt_mask = query_mask
        query_mask[
            torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=self.device), 2)
        ] = 0
        if self.refine_mask == "onnx":
            # NDLOCR-Lite の ONNX では、このマスクが「値 1.0 の float 行列」として書き出されていて、遮蔽(-inf)
            # ではなく +1.0 の加算バイアスとして働く(自分自身のトークンが見える)。同梱 ONNX と数値一致させる。
            query_mask = query_mask.float()
            tgt_mask = tgt_mask.float()
        bos = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self.device)
        for _ in range(iters):
            tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
            tgt_padding_mask = (tgt_in == self.eos_id).int().cumsum(-1) > 0
            if self.refine_mask == "onnx":  # マスクの型を float に揃える(-inf = 遮蔽。計算は同じ)
                tgt_padding_mask = torch.zeros(
                    tgt_padding_mask.shape, device=self.device
                ).masked_fill(tgt_padding_mask, float("-inf"))
            tgt_out = self.decode(
                tgt_in,
                memory,
                tgt_mask,
                tgt_padding_mask,
                pos_queries,
                query_mask[:, : tgt_in.shape[1]],
            )
            logits = self._head(tgt_out, vocab_mask)
        return logits

    def next_logits(
        self, memory: Tensor, prefix: Tensor, vocab_mask: Tensor | None = None
    ) -> Tensor:
        """AR(前向き)で、プレフィックス `prefix`(N, j: BOS + j-1 文字)の次の 1 トークンの logits(N, V)。

        メモリ(エンコーダ出力)は N と同じ数だけ複製して渡す(ビームサーチ用)。
        """
        n, j = prefix.shape
        mask = torch.triu(torch.ones((j, j), dtype=torch.bool, device=self.device), 1)
        pos_q = self.pos_queries[:, j - 1 : j].expand(n, -1, -1)
        out = self.decode(prefix, memory, mask, tgt_query=pos_q, tgt_query_mask=mask[j - 1 : j, :j])
        return self._head(out, vocab_mask).squeeze(1)

    def beam_search(
        self, memory: Tensor, beam: int, vocab_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """AR(前向き)のビームサーチ。戻り値: (tokens (N, K, L), 対数尤度 (N, K))、K 本を尤度の降順で。

        tokens は BOS を除く出力(EOS 以降は EOS で埋める)。対数尤度は文字 + 終端 EOS の合計(マスク適用後に
        正規化した分布で、`token_logprobs` と同じ定義)。最大長(max_label_length 文字)では EOS を強制する。
        """
        n = memory.shape[0]
        k = beam
        v = self.head.out_features
        mem = [
            (kk.repeat_interleave(k, dim=0), vv.repeat_interleave(k, dim=0))
            for kk, vv in self.project_memory(memory)
        ]
        vmask = None if vocab_mask is None else (
            vocab_mask.repeat_interleave(k, dim=0) if vocab_mask.dim() == 2 else vocab_mask
        )  # fmt: skip
        seqs = torch.full((n * k, 1), self.bos_id, dtype=torch.long, device=self.device)
        scores = torch.zeros(n, k, device=self.device)
        scores[:, 1:] = float("-inf")  # 最初は 1 本だけ生かす(同じ候補の重複を避ける)
        finished = torch.zeros(n * k, dtype=torch.bool, device=self.device)
        neg_inf = float("-inf")
        ar = torch.arange(n, device=self.device)[:, None]
        for step in range(self.max_label_length + 1):
            logp = torch.log_softmax(self.next_logits(mem, seqs, vmask).float(), dim=-1)
            if step == self.max_label_length:  # これ以上は文字を足せない
                forced = torch.full_like(logp, neg_inf)
                forced[:, self.eos_id] = logp[:, self.eos_id].clamp(min=-30.0)
                logp = forced
            if finished.any():
                done = torch.full_like(logp, neg_inf)
                done[:, self.eos_id] = 0.0
                logp = torch.where(finished[:, None], done, logp)
            total = (scores.reshape(n * k, 1) + logp).reshape(n, k * v)
            scores, idx = total.topk(k, dim=-1)
            src = idx // v
            tok = idx % v
            flat = (ar * k + src).reshape(-1)
            seqs = torch.cat([seqs[flat], tok.reshape(-1, 1)], dim=1)
            finished = finished[flat] | (tok.reshape(-1) == self.eos_id)
            if bool(finished.all()):
                break
        out = seqs[:, 1:].reshape(n, k, -1)
        return out, scores

    def token_logprobs(
        self, memory: Tensor | list, token_ids: list[list[int]], vocab_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """教師強制(AR・前向き)での、出力トークン(文字 + 終端 EOS)ごとの対数確率 (N, L) と有効位置 (N, L)。

        token_ids は EOS を含まない文字 ID。memory は (1 または N, T, C) か、`project_memory` の結果。
        マスクで許可されない文字を含む系列は -inf になる。
        """
        n = len(token_ids)
        max_len = max((len(t) for t in token_ids), default=0)
        if max_len > self.max_label_length:
            raise ValueError("文字列が長すぎます")
        tgt = torch.full((n, max_len + 1), self.pad_id, dtype=torch.long, device=self.device)
        tgt[:, 0] = self.bos_id
        out_ids = torch.full((n, max_len + 1), self.pad_id, dtype=torch.long, device=self.device)
        for k, ids in enumerate(token_ids):
            if ids:
                tgt[k, 1 : 1 + len(ids)] = torch.as_tensor(
                    ids, dtype=torch.long, device=self.device
                )
                out_ids[k, : len(ids)] = tgt[k, 1 : 1 + len(ids)]
            out_ids[k, len(ids)] = self.eos_id
        length = max_len + 1
        mask = torch.triu(torch.ones((length, length), dtype=torch.bool, device=self.device), 1)
        out = self.decode(tgt, memory, mask, tgt_query_mask=mask)
        logits = self._head(out, vocab_mask)
        logp = torch.log_softmax(logits, dim=-1)
        valid = out_ids != self.pad_id
        safe_ids = out_ids.clamp(max=logp.shape[-1] - 1)
        picked = logp.gather(-1, safe_ids.unsqueeze(-1)).squeeze(-1)
        picked = torch.where(valid, picked, torch.zeros_like(picked))
        return picked, valid

    def score_logprob(self, memory: Tensor, token_ids: list[list[int]], vocab_mask=None) -> Tensor:
        """系列の対数尤度(文字 + 終端 EOS)。(N, 2)… [合計の対数尤度, 文字数 + 1]。"""
        picked, valid = self.token_logprobs(memory, token_ids, vocab_mask)
        return torch.stack([picked.sum(-1), valid.sum(-1).to(picked.dtype)], dim=-1)
